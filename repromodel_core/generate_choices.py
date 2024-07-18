from timm import optim as timm_optim
from timm import loss as timm_loss

from torch import optim
from torch.nn.modules import loss
from torch.optim import lr_scheduler

from typing import Union, get_type_hints, Literal, Optional, List, Dict, Any

import ast
import inspect
import json
import os
import pkgutil

import torch
import torchmetrics
import torchvision

######################################################################
# HELPER FUNCTIONS
######################################################################

# Function to safely evaluate AST literals to their respective Python types.
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
        if node.id in ['None', 'True', 'False']:
            return eval(node.id)
        elif node.id in ['float', 'int', 'str', 'bool', 'Compose']:
            return node.id
        else:
            return node.id
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['float', 'int', 'str', 'bool', 'Compose']:
                return eval(node.func.id)(*[evaluate_ast_literal(arg) for arg in node.args])
            elif node.func.id == 'type':
                return type(*[evaluate_ast_literal(arg) for arg in node.args])
        return "<function>"
    elif isinstance(node, ast.Lambda):
        return eval(compile(ast.Expression(node), '<string>', 'eval'))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -evaluate_ast_literal(node.operand)
        else:
            return evaluate_ast_literal(node.operand)
    else:
        return "<unsupported>"

# Function to parse the @enforce_types_and_ranges decorator for types, default values, ranges, and options.
# Example: @enforce_types_and_ranges({ 'weight': { 'type': float, 'range': (0.1, 1.0) })
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

# Function to parse the @tag decorator for task, subtask, modality, and submodality.
# Example: @tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
def parse_tag_decorator(decorator):
    tags = {}
    if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id') and decorator.func.id == 'tag':
        for keyword in decorator.keywords:
            if isinstance(keyword, ast.keyword):
                key = keyword.arg
                value = evaluate_ast_literal(keyword.value)
                if isinstance(value, list):
                    tags[key] = value
                else:
                    tags[key] = [value]
    return tags

# Function to parse Python files and extract classes, their __init__ parameters, and tags.
def parse_python_file(file_path):
    with open(file_path, 'r') as file:
        node = ast.parse(file.read(), filename=file_path)
    class_definitions = {}
    class_tags = {}
    filename = os.path.splitext(os.path.basename(file_path))[0]
    for n in node.body:
        if isinstance(n, ast.ClassDef) and not n.name.startswith('_'):
            params = {}
            tags = {}
            for item in n.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    for decorator in item.decorator_list:
                        params.update(parse_decorator(decorator))
            for decorator in n.decorator_list:
                tags.update(parse_tag_decorator(decorator))
            class_definitions[n.name] = params
            if tags:
                class_tags[n.name] = {'tags': tags, 'filename': filename}
    return class_definitions, class_tags

# Function to format type annotations into a more readable form.
def format_type(annotation, default=None):
    if isinstance(annotation, type):
        return annotation.__name__
    if hasattr(annotation, '__origin__'):
        if annotation.__origin__ is Union:
            valid_types = [arg for arg in annotation.__args__ if arg is not type(None)]
            if default is not None:
                for valid_type in valid_types:
                    if isinstance(default, valid_type):
                        return valid_type.__name__
            return format_type(valid_types[0])
        if annotation.__origin__ is Literal or 'Literal' in str(annotation):
            return "str"
        if annotation.__origin__ is Optional:
            return format_type(annotation.__args__[0])  # Handle Optional by using the inner type
        base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else repr(annotation.__origin__).split(' ')[0].split('.')[-1]
        if hasattr(annotation, '__args__'):
            args = ", ".join([format_type(arg) for arg in annotation.__args__])
            return f"{base}[{args}]"
        else:
            return base
    elif hasattr(annotation, '__args__'):
        args = ", ".join([format_type(arg) for arg in annotation.__args__])
        base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else str(annotation)
        return f"{base}[{args}]"
    elif hasattr(annotation, '__name__'):
        return annotation.__name__
    else:
        return str(annotation)
    
# Function to parse parameter details.
def parse_parameter_details(param):
    details = {}
    if param.annotation != inspect.Parameter.empty:
        details['type'] = format_type(param.annotation, param.default)
        if 'Literal' in str(param.annotation) and hasattr(param.annotation, '__args__'):
            details['options'] = [arg for arg in param.annotation.__args__]
    if param.default != inspect.Parameter.empty:
        details['default'] = param.default
    return details

# Function to extract class definitions with __init__ parameters.
def extract_classes_with_init_params(module, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    classes = {}

    def format_type(annotation, default=None):
        if isinstance(annotation, type):
            return annotation.__name__
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is Union:
                valid_types = [arg for arg in annotation.__args__ if arg is not type(None)]
                if default is not None:
                    default_type = type(default)
                    for valid_type in valid_types:
                        if isinstance(default, valid_type):
                            return valid_type.__name__
                return format_type(valid_types[0])
            if annotation.__origin__ is Literal or 'Literal' in str(annotation):
                return "str"
            if annotation.__origin__ is Optional:
                return format_type(annotation.__args__[0])  # Handle Optional by using the inner type
            base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else repr(annotation.__origin__).split(' ')[0].split('.')[-1]
            if hasattr(annotation, '__args__'):
                args = ", ".join([format_type(arg) for arg in annotation.__args__])
                return f"{base}[{args}]"
            else:
                return base
        elif hasattr(annotation, '__args__'):
            args = ", ".join([format_type(arg) for arg in annotation.__args__])
            base = annotation.__origin__.__name__ if hasattr(annotation.__origin__, '__name__') else str(annotation)
            return f"{base}[{args}]"
        elif hasattr(annotation, '__name__'):
            return annotation.__name__
        else:
            return str(annotation)

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
                                    if param.default == 0:
                                        param_type = "int, float"
                                    else:
                                        param_type = type(param.default).__name__
                            elif hasattr(param_type, '__origin__'):
                                if param_type.__origin__ is Union:
                                    valid_types = [arg for arg in param_type.__args__ if arg is not type(None)]
                                    if param.default is not inspect.Parameter.empty:
                                        default_type = type(param.default)
                                        for valid_type in valid_types:
                                            if isinstance(param.default, valid_type):
                                                param_type = valid_type
                                                break
                                    else:
                                        param_type = valid_types[0]
                                elif param_type.__origin__ is Optional:
                                    param_type = param_type.__args__[0]
                        except Exception as e:
                            print(f"Failed to get type hint for {param_name} in {name}: {e}")
                            param_type = str(param.annotation)
                        param_info[param_name] = {
                            "type": format_type(param_type) if param_type is not inspect._empty else "unknown",
                            "default": param.default if param.default is not inspect.Parameter.empty else None
                        }
                    classes[name] = param_info
                    if not param_info:
                        print(f"No parameters found for {name}'s __init__ method")
                else:
                    print(f"No __init__ method found for {name}")

        if hasattr(mod, '__path__'):
            for importer, submodname, ispkg in pkgutil.iter_modules(mod.__path__):
                full_submodname = f"{mod.__name__}.{submodname}"
                submod = __import__(full_submodname, fromlist=[submodname])
                inspect_module(submod)

    inspect_module(module)
    return classes

# Function to make object serializable.
def make_json_serializable(obj, leading_key=None):
    if isinstance(obj, dict):
        if leading_key is None:
            return {k.split('>')[-1]: make_json_serializable(v) for k, v in obj.items()}
        else:
            return {k.split('>')[-1]: make_json_serializable(v) for k, v in obj.items()}
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

# Function that returns devices.
def get_devices():
    device_definitions = {
        "type": "str",
        "default": "cpu",
        "options": ['cpu']
    }
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            cuda_key = f'cuda:{i}'
            device_definitions["options"].append(cuda_key)
    
    device_definitions["options"] = str(device_definitions["options"])
    return device_definitions

######################################################################
# CONSTANTS
######################################################################

base_path = "repromodel_core/src/"

tags_structure = { "tags_per_class": {}, "class_per_tag": { "task": {}, "subtask": {}, "modality": {}, "submodality": {} } }

directories = [
    "models", "preprocessing", "datasets", "augmentations",
    "metrics", "losses", "early_stopping", "postprocessing"
]

loss_classes = [
    "L1Loss", "MSELoss", "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss",
    "GaussianNLLLoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss",
    "HingeEmbeddingLoss", "MultiLabelMarginLoss", "HuberLoss", "SmoothL1Loss", "SoftMarginLoss",
    "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss", "MultiMarginLoss", "TripletMarginLoss",
    "TripletMarginWithDistanceLoss"
]

optimizer_classes = [
    "Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax", "ASGD", "LBFGS",
    "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"
]

######################################################################
# MAIN METHOD
######################################################################

if __name__ == "__main__":

    # Construct choices.json file by collecting all definitions from specified directories.
    json_obj = {}

    ######################################################################
    # Key: load_from_checkpoint
    # Description:
    ######################################################################

    json_obj["load_from_checkpoint"] = {
        "type": "bool",
        "default": False
    }

    ########################################################################################################################
    # Key: ["models", "preprocessing", "datasets", "augmentations", "metrics", "losses", "early_stopping", "postprocessing"]
    ########################################################################################################################

    for directory in directories:
        full_path = os.path.join(base_path, directory)
        directory_definitions = {}
        for root, dirs, files in os.walk(full_path, topdown=True):
            dirs[:] = [d for d in dirs if '.ipynb_checkpoints' not in d]
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    definitions, tags = parse_python_file(file_path)
                    if definitions:
                        file_name_without_extension = os.path.splitext(file)[0]
                        directory_definitions[file_name_without_extension] = definitions
                    if tags:
                        for class_name, class_info in tags.items():
                            # Add to tags_per_class.
                            full_class_name = f"{class_info['filename']}>{class_name}"
                            tags_structure["tags_per_class"][full_class_name] = class_info['tags']
                            
                            # Add to class_per_tag.
                            for key, values in class_info['tags'].items():
                                for value in values:
                                    if value not in tags_structure["class_per_tag"][key]:
                                        tags_structure["class_per_tag"][key][value] = {}
                                    if directory not in tags_structure["class_per_tag"][key][value]:
                                        tags_structure["class_per_tag"][key][value][directory] = []
                                    tags_structure["class_per_tag"][key][value][directory].append(full_class_name)
        if directory_definitions:
            json_obj[directory] = directory_definitions

    # Convert sets to lists for JSON serialization.
    for key, value in tags_structure["class_per_tag"].items():
        for subkey, subvalue in value.items():
            for directory, models in subvalue.items():
                tags_structure["class_per_tag"][key][subkey][directory] = list(models)

    for key, value in tags_structure["tags_per_class"].items():
        for subkey, subvalue in value.items():
            if isinstance(subvalue, set):
                tags_structure["tags_per_class"][key][subkey] = list(subvalue)
            elif isinstance(subvalue, dict):
                tags_structure["tags_per_class"][key][subkey] = {k: list(v) if isinstance(v, set) else v for k, v in subvalue.items()}
    
    # Key: augmentations  >  torchvision>transforms
    # Description: Extract torchvision augmentations.
    all_tv_augs = extract_classes_with_init_params(torchvision.transforms)
    leading_key = 'torchvision>transforms'
    json_obj['augmentations'][leading_key] = make_json_serializable(all_tv_augs, leading_key=leading_key)

    # Key: metrics  >  torchmetrics
    # Description: Extract torchmetric classes.
    all_torchmetrics = extract_classes_with_init_params(torchmetrics)
    leading_key = 'torchmetrics'
    json_obj['metrics'][leading_key] = make_json_serializable(all_torchmetrics, leading_key=leading_key)

    # Key: losses  >  torch>nn>modules>loss
    # Description: Extract classes from torch.nn for the specified loss classes.
    loss_classes_extracted = extract_classes_with_init_params(loss, loss_classes)
    leading_key = 'torch>nn>modules>loss'
    json_obj['losses'][leading_key] = make_json_serializable(loss_classes_extracted, leading_key=leading_key)

    # Key: losses  >  timm>loss
    # Description: Extract classes from torch.nn for the specified loss classes.
    timm_loss_classes_extracted = extract_classes_with_init_params(timm_loss)
    leading_key = 'timm>loss'
    json_obj['losses'][leading_key] = make_json_serializable(timm_loss_classes_extracted, leading_key=leading_key)

    ######################################################################
    # Key: tags
    # Description: The associated tags for each Python class.
    ######################################################################
    json_obj["tags"] = tags_structure

    ######################################################################
    # Key: lr_schedulers
    # Description: Various learning rate schedulers.
    ######################################################################

    # Extract classes from torch.optim.lr_scheduler and add to all_classes
    lr_scheduler_classes = extract_classes_with_init_params(lr_scheduler)
    json_obj['lr_schedulers'] = {}
    leading_key = 'torch>optim>lr_scheduler'
    json_obj['lr_schedulers'][leading_key] = make_json_serializable(lr_scheduler_classes)

    ######################################################################
    # Key: optimizers
    # Description: Various optimization algorithms.
    ######################################################################

    # Key: optimizers  >  torch>optim
    # Extract classes from torch.optim for the specified optimizer classes
    optimizer_classes_extracted = extract_classes_with_init_params(optim, optimizer_classes)
    json_obj['optimizers'] = {}
    leading_key = 'torch>optim'
    json_obj['optimizers'][leading_key] = make_json_serializable(optimizer_classes_extracted, leading_key=leading_key)

    # Key: optimizers  >  timm>optim
    # Extract classes from timm.optimizers
    timm_optimizer_classes_extracted = extract_classes_with_init_params(timm_optim)
    leading_key = 'timm>optim'
    json_obj['optimizers'][leading_key] = make_json_serializable(timm_optimizer_classes_extracted, leading_key=leading_key)

    ######################################################################
    # Key: batch_size
    # Description: The batch size for training.
    ######################################################################
    
    json_obj["batch_size"] = {
        "type": "int",
        "default": 1,
        "range": "(1, 1024)"
    }

    ######################################################################
    # Key: monitor
    # Description: Monitor the performance of the model.
    ######################################################################

    json_obj["monitor"] = {
        "type": "str",
        "default": "val_loss",
        "options": "['train_loss', 'val_loss']"
    }

    ######################################################################
    # Key: data_splits
    # Description:
    ######################################################################

    json_obj["data_splits"] = {
        "k": {
            "type": "int",
            "default": 5,
            "range": "(1, 20)"
        },
        "random_seed": {
            "type": "int",
        }
    }

    ######################################################################
    # Key: model_save_path
    # Description: Output location for model.
    ######################################################################

    json_obj["model_save_path"] = {
        "type": "str",
        "default": "repromodel_core/ckpts/"
    }

    ######################################################################
    # Key: tensorboard_log_path
    # Description: Output location of Tensorboard logs.
    ######################################################################

    json_obj["tensorboard_log_path"] = {
        "type": "str",
        "default": "repromodel_core/logs"
    }

    ######################################################################
    # Key: progress_path
    # Description: Output location of progress.json.
    ######################################################################

    json_obj["progress_path"] = {
        "type": "str",
        "default": "repromodel_core/ckpts/progress.json"
    }

    ######################################################################
    # Key: training_name
    # Description:
    ######################################################################

    json_obj["training_name"] = {
        "type": "str"
    }

    ######################################################################
    # Key: device
    # Description:
    ######################################################################

    json_obj["device"] = get_devices()

    ######################################################################
    # Convert Sets to Lists
    ######################################################################
    
    for key, value in tags_structure["class_per_tag"].items():
        for subkey, subvalue in value.items():
            for directory, models in subvalue.items():
                tags_structure["class_per_tag"][key][subkey][directory] = list(models)

    for key, value in tags_structure["tags_per_class"].items():
        for subkey, subvalue in value.items():
            if isinstance(subvalue, set):
                tags_structure["tags_per_class"][key][subkey] = list(subvalue)
            elif isinstance(subvalue, dict):
                tags_structure["tags_per_class"][key][subkey] = { k: list(v) if isinstance(v, set) else v for k, v in subvalue.items() }

    # Add tags_structure under "tags" key.
    json_obj["tags"] = tags_structure

    # Merge tag_definitions with json_obj.
    json_obj.update({"tags": tags_structure})

    ######################################################################
    # Write JSON Object to File
    ######################################################################
    with open('repromodel_core/choices.json', 'w') as json_file:
        json.dump(json_obj, json_file, indent=4, default=str)

    print("Class definitions with __init__ parameters, types, default values, and tags have been extracted and grouped by directory in choices.json")