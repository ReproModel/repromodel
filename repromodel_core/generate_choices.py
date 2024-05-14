import os
import ast
import json

# Base path for your project
base_path = 'src/'

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

# Collect all definitions from specified directories
all_definitions = {}

all_definitions["load_from_checkpoint"] = {
                        "type": "bool",
                        "default": False
                    }

for directory in ['models', 'datasets', 'metrics', 'preprocessing', 'postprocessing', 'losses', 'augmentations', 'early_stopping']:
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

all_definitions["val_loss"] = {
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
with open('choices.json', 'w') as json_file:
    json.dump(all_definitions, json_file, indent=4)

print("Class definitions with __init__ parameters, types, and default values have been extracted and grouped by directory in choices.json")
