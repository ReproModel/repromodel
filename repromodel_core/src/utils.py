import os
import json
import ast
import torch
from datetime import datetime

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

def print_to_file(string, config = None, tqdm=False, model_num = None):
    """Print a string to a file, with optional tqdm compatibility."""
    if config is not None:
        file_name = f"{config.tensorboard_log_path}/{config.model[model_num]}_{config.dataset}_" + "command_output.txt"
    else:
        file_name = "repromodel_core/logs/command_output.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Open the file in write mode if tqdm is True, otherwise append
    mode = "w" if tqdm else "a"
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the string with the timestamp
    timestamped_string = f"[{timestamp}] {string}"

    with open(file_name, mode) as file:
        if tqdm:
            file.write(timestamped_string)  
        else:
            file.write(timestamped_string + '\n')
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

def delete_command_outputs():
    file_path = "repromodel_core/logs/command_output.txt"

    try:
        os.remove(file_path)
        print_to_file(f"Command output {file_path} successfully restarted.")
    except FileNotFoundError:
        print_to_file(f"Old command output file {file_path} not found.")
    except PermissionError:
        print_to_file(f"Permission denied to delete {file_path}.")
    except Exception as e:
        print_to_file(f"Error occurred while deleting {file_path}: {e}")

class TqdmFile:
    def __init__(self, config=None, model_num = None):
        self.config = config
        self.model_num = model_num

    def write(self, msg):
        # Append to the file in a TQDM-compatible way
        print_to_file(msg, config=self.config, tqdm=True, model_num=self.model_num)

    def flush(self):
        pass  # No-op to conform to file interface

