import os
import ast
import json
import torchmetrics
import inspect
import pkgutil
import importlib
from torch.optim import lr_scheduler
from typing import Union, get_type_hints, Literal, Optional, List
from torch.nn.modules import loss

# Base path for your project
base_path = 'repromodel_core/src/'

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
                                if k.s in ['type', 'default', 'range', 'options']:
                                    properties[k.s] = ast.unparse(v)
                    param_info[key.s] = properties
    return param_info

# Function to parse Python files and extract classes and their __init__ parameters
def parse_python_file(file_path):
    with open(file_path, 'r') as file:
        node = ast.parse(file.read(), filename=file_path)
    class_definitions = {}
    for n in node.body:
        if isinstance(n, ast.ClassDef):
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

def find_functions_in_submodules(module, prefix=""):
    """
    Recursively finds all functions in the submodules of the given module, including nested submodules.
    """
    function_dict = {}
    if hasattr(module, "__path__"):
        for _, modname, ispkg in pkgutil.iter_modules(module.__path__):
            submodule = importlib.import_module(module.__name__ + "." + modname)
            for name, obj in inspect.getmembers(submodule):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    try:
                        params = get_type_hints(obj, globalns=vars(submodule))
                    except Exception as e:
                        print(f"Could not get type hints for function {name} in module {submodule}: {e}")
                        params = {}
                    formatted_params = {
                        p: parse_parameter_details(inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=params.get(p)))
                        for p in obj.__code__.co_varnames
                        if p in params and p != 'kwargs'
                    }
                    function_dict[prefix + modname + "." + name] = formatted_params
            # Recursive call to handle further nested submodules
            nested_functions = find_functions_in_submodules(submodule, prefix + modname + ".")
            function_dict.update(nested_functions)
    return function_dict

# Function to extract class definitions with __init__ parameters
def extract_classes_with_init_params(module, class_names: Optional[List[str]] = None):
    classes = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and not name.startswith('_'):
            if class_names and name not in class_names:
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

def create_functions_json(module_name):
    module = importlib.import_module(module_name)
    all_functions = find_functions_in_submodules(module)
    serializable_functions = make_json_serializable(all_functions)
    return serializable_functions
    
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

all_torchmetrics = find_functions_in_submodules(torchmetrics)
serializable_torchmetrics = make_json_serializable(all_torchmetrics)
all_definitions['metrics']['torchmetrics'] = serializable_torchmetrics

# Extract classes from torch.optim.lr_scheduler and add to all_classes
lr_scheduler_classes = extract_classes_with_init_params(lr_scheduler)
all_definitions['lr_schedulers'] = lr_scheduler_classes

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

# Static choices 
all_definitions["device"] = {
                    "type": "str",
                    "default": "cpu",
                    "options": "['cpu', 'cuda']"
                }

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
                        "type": "str"
                    }


all_definitions["tensorboard_log_path"] = {
                        "type": "str",
                        "default": "logs"
                    }

all_definitions["metadata_path"] = {
                        "type": "str",
                        "default": "logs/metadata.json"
                    }

all_definitions["training_name"] = {
                        "type": "str",
                       
                    }

# Save the collected data to a JSON file
with open('repromodel_core/choices.json', 'w') as json_file:
    json.dump(all_definitions, json_file, indent=4)

print("Class definitions with __init__ parameters, types, and default values have been extracted and grouped by directory in choices.json")
