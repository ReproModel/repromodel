import os
import ast
import json
import torchmetrics
import inspect
import pkgutil
import torch
import platform
from torch.optim import lr_scheduler
from typing import Union, get_type_hints, Literal, Optional, List, Dict, Any
from torch.nn.modules import loss
from torch import optim

# Base path for your project
base_path = 'repromodel_core/src/'

# Function to safely evaluate AST literals to their respective Python types
def evaluate_ast_literal(node):
    if isinstance(node, ast.Constant):  # For Python 3.8+
        return node.value
    elif isinstance(node, (ast.Str, ast.Num, ast.NameConstant)):  # For older versions of Python
        return node.n if isinstance(node, ast.Num) else node.s if isinstance(node, ast.Str) else node.value
    elif isinstance(node, ast.Tuple):
        return tuple(evaluate_ast_literal(el) for el in node.elts)
    elif isinstance(node, ast.List):
        return [evaluate_ast_literal(el) for el in node.elts]
    elif isinstance(node, ast.Dict):
        return {evaluate_ast_literal(k): evaluate_ast_literal(v) for k, v in zip(node.keys, node.values)}
    elif isinstance(node, ast.Name):
        # Handle the case where the value is a named constant or type, e.g., `None`, `True`, `False`, `float`, `int`
        if node.id in ['None', 'True', 'False']:
            return eval(node.id)
        elif node.id in ['float', 'int', 'str', 'bool', 'Compose']:
            return node.id
        else:
            return node.id  # Return the name as a string if it's not a recognized built-in type
    elif isinstance(node, ast.Call):
        # Handle function calls, assume they return literals or types for simplicity
        if isinstance(node.func, ast.Name):
            if node.func.id in ['float', 'int', 'str', 'bool', 'Compose']:
                return eval(node.func.id)(*[evaluate_ast_literal(arg) for arg in node.args])
            elif node.func.id == 'type':
                # Handle type() function call
                return type(*[evaluate_ast_literal(arg) for arg in node.args])
        return "<function>"  # Return a placeholder for function calls
    elif isinstance(node, ast.Lambda):
        # Handle lambda functions
        return eval(compile(ast.Expression(node), '<string>', 'eval'))
    elif isinstance(node, ast.UnaryOp):
        # Handle unary operations (e.g., -1)
        if isinstance(node.op, ast.USub):
            return -evaluate_ast_literal(node.operand)
        else:
            return evaluate_ast_literal(node.operand)
    else:
        return "<unsupported>"  # Generic placeholder for unsupported node types

# Function to parse the decorator for types, default values, ranges, and options
def parse_decorator(decorator):
    param_info = {}
    if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id') and decorator.func.id == 'enforce_types_and_ranges':
        if isinstance(decorator.args[0], ast.Dict):
            for key, val in zip(decorator.args[0].keys, decorator.args[0].values):
                if isinstance(key, ast.Str):
                    properties = {}
                    if isinstance(val, ast.Dict):
                        for k, v in zip(val.keys, val.values):
                            if isinstance(k, ast.Str):
                                if k.s in ['type', 'default']:
                                    evaluated_value = evaluate_ast_literal(v)
                                    properties[k.s] = evaluated_value
                                elif k.s in ['range', 'options']:
                                    evaluated_value = evaluate_ast_literal(v)
                                    properties[k.s] = str(evaluated_value)
                    param_info[key.s] = properties
    return param_info

# Function to parse Python files and extract classes and their __init__ parameters
def parse_python_file(file_path):
    with open(file_path, 'r') as file:
        node = ast.parse(file.read(), filename=file_path)
    class_definitions = {}
    for n in node.body:
        if isinstance(n, ast.ClassDef) and not n.name.startswith('_'):
            for item in n.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    params = {}
                    for decorator in item.decorator_list:
                        params.update(parse_decorator(decorator))
                    class_definitions[n.name] = params
    return class_definitions

def format_type(annotation):
    """
    Format complex type annotations into a more readable form, handling Union and Literal types.
    """
    if isinstance(annotation, type):
        return annotation.__name__
    if hasattr(annotation, '__origin__'):
        if annotation.__origin__ is Union:
            valid_types = [arg for arg in annotation.__args__ if arg is not type(None)]
            return ", ".join(format_type(t) for t in valid_types)  # Simplify Union types
        if annotation.__origin__ is Literal or 'Literal' in str(annotation):
            return "str"  # Simplify Literal types to 'str'
        base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else repr(annotation.__origin__).split(' ')[0].split('.')[-1]
        if hasattr(annotation, '__args__'):
            args = ", ".join([format_type(arg) for arg in annotation.__args__])
            return f"{base}[{args}]"
        else:
            return base
    elif hasattr(annotation, '__args__'):
        # Handle generic types that are not covered above (e.g., List[int])
        args = ", ".join([format_type(arg) for arg in annotation.__args__])
        base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else str(annotation)
        return f"{base}[{args}]"
    elif hasattr(annotation, '__name__'):
        return annotation.__name__
    else:
        return str(annotation)  # Fallback for unhandled types

def parse_parameter_details(param):
    """
    Parse details of a parameter including type, default (if not None), and options if Literal.
    """
    details = {}
    if param.annotation != inspect.Parameter.empty:
        details['type'] = format_type(param.annotation)
        if 'Literal' in str(param.annotation) and hasattr(param.annotation, '__args__'):
            details['options'] = [arg for arg in param.annotation.__args__]
    if param.default != inspect.Parameter.empty:
        details['default'] = param.default  # Capture default values for all parameters, including Unions
    return details

# Function to extract class definitions with __init__ parameters
def extract_classes_with_init_params(module, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    classes = {}

    def inspect_module(mod):
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__.startswith(mod.__name__) and not name.startswith('_'):
                if class_names and name not in class_names:
                    continue
                if name.startswith('_'):
                    continue
                init = obj.__init__
                if init:
                    params = inspect.signature(init).parameters
                    param_info = {}
                    for param_name, param in params.items():    
                        if param_name == 'self' or param_name == 'optimizer':
                            continue
                        try:
                            param_type = get_type_hints(init).get(param_name, param.annotation)
                            if param_type is inspect._empty:
                                if param.default is not inspect.Parameter.empty:
                                    # Attempt to infer the type from the default value
                                    if param.default == 0:
                                        param_type = "int, float"
                                    else:
                                        param_type = type(param.default).__name__
                            elif isinstance(param_type, type):
                                param_type = param_type.__name__
                        except Exception as e:
                            print(f"Failed to get type hint for {param_name} in {name}: {e}")
                            param_type = str(param.annotation)
                        param_info[param_name] = {
                            "type": param_type if param_type is not inspect._empty else "unknown",
                            "default": param.default if param.default is not inspect.Parameter.empty else None
                        }
                    classes[name] = param_info
                    if not param_info:
                        print(f"No parameters found for {name}'s __init__ method")
                else:
                    print(f"No __init__ method found for {name}")

        # Recursively inspect submodules if the module has a __path__ attribute
        if hasattr(mod, '__path__'):
            for importer, submodname, ispkg in pkgutil.iter_modules(mod.__path__):
                full_submodname = f"{mod.__name__}.{submodname}"
                submod = __import__(full_submodname, fromlist=[submodname])
                inspect_module(submod)

    inspect_module(module)
    return classes

def make_json_serializable(obj):
    """
    Recursively convert non-serializable objects to their string representations.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, set):
        return {make_json_serializable(i) for i in obj}
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
        
def get_devices():
    device_definitions = {
        "type": "str",
        "default": "cpu",
        "options": "['cpu']"
    }
    
    # Check if CUDA is available and add CUDA devices
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            cuda_key = 'cuda:{}'.format(i)
            device_definitions["options"].append(cuda_key)
    
    # Convert options list to string format    
    return device_definitions

# Collect all definitions from specified directories
all_definitions = {}

all_definitions["load_from_checkpoint"] = {
                        "type": "bool",
                        "default": False
                    }
for directory in ['models', 'preprocessing', 'datasets', 'augmentations', 'metrics', 'losses', 'early_stopping', 'postprocessing']:
    full_path = os.path.join(base_path, directory)
    directory_definitions = {}
    for root, dirs, files in os.walk(full_path, topdown=True):
        dirs[:] = [d for d in dirs if '.ipynb_checkpoints' not in d]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                definitions = parse_python_file(file_path)
                if definitions:
                    file_name_without_extension = os.path.splitext(file)[0]
                    directory_definitions[file_name_without_extension] = definitions
    if directory_definitions:
        all_definitions[directory] = directory_definitions

all_torchmetrics = extract_classes_with_init_params(torchmetrics)
all_definitions['metrics']['torchmetrics'] = make_json_serializable(all_torchmetrics)

# Extract classes from torch.optim.lr_scheduler and add to all_classes
lr_scheduler_classes = extract_classes_with_init_params(lr_scheduler)
all_definitions['lr_schedulers'] = {}
all_definitions['lr_schedulers']['torch.optim'] = make_json_serializable(lr_scheduler_classes)

loss_classes = [
    "L1Loss", "MSELoss", "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss",
    "GaussianNLLLoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss",
    "HingeEmbeddingLoss", "MultiLabelMarginLoss", "HuberLoss", "SmoothL1Loss", "SoftMarginLoss",
    "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss", "MultiMarginLoss", "TripletMarginLoss",
    "TripletMarginWithDistanceLoss"
]

# Extract classes from torch.nn for the specified loss classes
loss_classes_extracted = extract_classes_with_init_params(loss, loss_classes)
all_definitions['losses']['torch.nn.modules.loss'] = make_json_serializable(loss_classes_extracted)

# List of specific class names to extract from torch.optim
optimizer_classes = [
    "Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax", "ASGD", "LBFGS",
    "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"
]

# Extract classes from torch.nn for the specified loss classes
optimizer_classes_extracted = extract_classes_with_init_params(optim, optimizer_classes)
all_definitions['optimizers'] = {}
all_definitions['optimizers']['torch.optim'] = make_json_serializable(optimizer_classes_extracted) #make_json_serializable(optimizer_classes_extracted)

# Static choices 

all_definitions["batch_size"] = {
                    "type": "int",
                    "default": 1,
                    "range": "(1, 1024)"
                }

all_definitions["monitor"] = {
                    "type": "str",
                    "default": "val_loss",
                    "options": "['train_loss', 'val_loss']"
                }

all_definitions["data_splits"] = {
                    "k": {
                        "type": "int",
                        "default": 5,
                        "range": "(1, 20)"
                    },
                    "random_seed": {
                        "type": "int",
                    },
                }

all_definitions["model_save_path"] = {
                        "type": "str",
                        "default": "repromodel_core/ckpts/"
                    }

all_definitions["tensorboard_log_path"] = {
                        "type": "str",
                        "default": "repromodel_core/logs"
                    }

all_definitions["progress_path"] = {
                        "type": "str",
                        "default": "repromodel_core/ckpts/progress.json"
                    }

all_definitions["training_name"] = {
                        "type": "str",
                    }

# choose device
all_definitions["device"] = get_devices()

# Save the collected data to a JSON file
with open('repromodel_core/choices.json', 'w') as json_file:
    json.dump(all_definitions, json_file, indent=4, default=str)

print("Class definitions with __init__ parameters, types, and default values have been extracted and grouped by directory in choices.json")
