import os
import json
import ast
import torch
from scipy.ndimage import zoom
from easydict import EasyDict as edict

def parse_constructor_params(node):
    """Extract constructor parameters and type annotations from a class node."""
    params = {}
    init_method = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == '__init__'), None)
    if init_method:
        for param in init_method.args.args:
            if param.arg != 'self':
                params[param.arg] = ast.unparse(param.annotation) if param.annotation else 'Any'
    return params

def get_classes_from_module(file_path, exclude_params={'transforms', 'mode', 'input_type'}):
    """Extract class definitions and their constructor parameters from a Python file."""
    classes = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        node = ast.parse(file.read())
        for n in node.body:
            if isinstance(n, ast.ClassDef) and "custom" not in n.name.lower():
                params = parse_constructor_params(n)
                for param in exclude_params:
                    params.pop(param, None)
                classes[n.name] = params
    return classes

def load_state(obj, state_dict):
    """Generic function to load a state dictionary into an object."""
    obj.load_state_dict(state_dict)
    return obj

def ensure_folder_exists(folder_path):
    """Ensure a folder exists, and if not, create it."""
    os.makedirs(folder_path, exist_ok=True)

def save_model(model, path, metadata=None):
    """Save a model and its metadata."""
    directory_path = os.path.dirname(path)
    ensure_folder_exists(directory_path)
    torch.save(model.state_dict(), path)
    if metadata:
        with open(path + '_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

def print_to_file(string, file_name="command_output.txt", tqdm=False):
    """Print a string to a file, with optional tqdm compatibility."""
    with open(file_name, "a") as file:
        if tqdm:
            file.write("\r" + string)
        else:
            file.write(string + "\n")
        file.flush()

# Example of refactoring save_model for different configurations
def handle_saving_logic(config, model, optimizer, lr_scheduler, path_prefix):
    """Handle the saving logic for different model configurations."""
    if config.model_type == "gan":
        save_model(model['generator'], f'{path_prefix}_generator.pt')
        save_model(model['discriminator'], f'{path_prefix}_discriminator.pt')
    else:
        save_model(model, f'{path_prefix}.pt')
    save_model(optimizer, f'{path_prefix}_optimizer.pt')
    save_model(lr_scheduler, f'{path_prefix}_lr_scheduler.pt')