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

def load_cfg(metadata_path):
    """
    Load a configuration and metadata from a metadata file.
    
    Args:
    - metadata_path (str): The path to the metadata file.

    Returns:
    - config (dict): The loaded configuration.
    - metadata (dict): The loaded metadata.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    config = metadata['config']
    print_to_file(f"Configuration and metadata {metadata_path} loaded")
    return config, metadata

def load_state(obj, state_dict):
    """Generic function to load a state dictionary into an object."""
    obj.load_state_dict(state_dict)
    return obj

def ensure_folder_exists(folder_path):
    """Ensure a folder exists, and if not, create it."""
    os.makedirs(folder_path, exist_ok=True)

# Example of refactoring save_model for different configurations
def handle_saving_state(model, model_name, optimizer, lr_scheduler, early_stopping, experiment_folder, suffix):
    # Save the model, optimizer, lr_scheduler, and early stopping state dictionaries
    torch.save(model.state_dict(), f'{experiment_folder}/{model_name}{suffix}.pt')
    torch.save(optimizer.state_dict(), f'{experiment_folder}/{model_name}{suffix}_optimizer.pt')
    torch.save(lr_scheduler.state_dict(), f'{experiment_folder}/{model_name}{suffix}_lr_scheduler.pt')
    if hasattr(early_stopping, 'state_dict'):
        torch.save(early_stopping.state_dict(), f'{experiment_folder}/{model_name}{suffix}_early_stopping.pt')

def get_last_dict_paths(model_save_path, model_name, fold):
    experiment_folder = model_save_path + model_name
    metadata_path = f"{experiment_folder}/{model_name}_best_fold_{fold}_metadata.json"
    _, metadata = load_cfg(metadata_path)
    paths = {}
    paths["model_path"] = metadata["model_state_dict_path"]
    paths["optimizer_path"] = metadata["optimizer_state_dict_path"]
    paths["scheduler_path"] = metadata["lr_scheduler_state_dict_path"]
    paths["es_path"] = metadata["early_stopping_state_dict_path"]
    return paths

def get_all_ckpts(model_save_path, models, num_folds):
    ckpts = {}
    for model_name in models:
        ckpts[model_name] = []
        experiment_folder = model_save_path + model_name
        for fold in range(num_folds):
            try:
                metadata_path = f"{experiment_folder}/{model_name}_best_fold_{fold}_metadata.json"
                _, metadata = load_cfg(metadata_path)
                ckpts[model_name].append(metadata["model_state_dict_path"])
            except FileNotFoundError:
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            except KeyError:
                raise KeyError(f"Key 'model_state_dict_path' is missing in metadata for {metadata_path}")
            except Exception as e:
                raise Exception(f"An unexpected error occurred: {str(e)}")
    return ckpts

def save_model(config, model, model_name, fold, epoch,
                optimizer, lr_scheduler, early_stopping, train_loss, val_loss, is_best=False):

    experiment_folder = config.model_save_path + model_name
    # Make sure that experiment folder exists
    ensure_folder_exists(experiment_folder)
                         
    suffix = f'_best_fold_{fold}' if is_best else f'_fold_{fold}_epoch_{epoch}'

    handle_saving_state(model, model_name, optimizer, lr_scheduler, early_stopping, experiment_folder, suffix)

    print_to_file(f"Saving model to {experiment_folder}/{model_name}{suffix}.pt")

    # Prepare metadata with conversion
    metadata = {
        'experiment_folder': experiment_folder,
        'model_name': model_name,
        'fold': fold,
        'epoch': epoch,
        'config': config,   
        'model_state_dict_path': f'{experiment_folder}/{model_name}{suffix}.pt',
        'optimizer_state_dict_path': f'{experiment_folder}/{model_name}{suffix}_optimizer.pt',
        'lr_scheduler_state_dict_path': f'{experiment_folder}/{model_name}{suffix}_lr_scheduler.pt',
        'early_stopping_state_dict_path': f'{experiment_folder}/{model_name}{suffix}_early_stopping.pt' if hasattr(early_stopping, 'state_dict') else '',
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    with open(f"{experiment_folder}/{model_name}{suffix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
        print_to_file(f"Metadata saved to {experiment_folder}/{model_name}{suffix}_metadata.json")

    progress = {
        'model_name': model_name,
        'fold': fold,
        'epoch': epoch
    }

    with open(config.progress_path, 'w') as f:
        json.dump(progress, f, indent=4)
        print_to_file(f"Progress saved to {config.progress_path}")    

def print_to_file(string, config = None, tqdm=False, model_num = None):
    """Print a string to a file, with optional tqdm compatibility."""
    if config is not None:
        file_name = f"{config.tensorboard_log_path}/{config.models[model_num]}_{config.datasets}_" + "command_output.txt"
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
        file.write(timestamped_string + ('\n' if not tqdm else ''))
        file.flush()

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

