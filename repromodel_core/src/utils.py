import os
import json
import ast
import shutil
import numpy as np
import torch
from datetime import datetime
import collections
from torchvision.models.inception import InceptionOutputs
from torchvision.models.googlenet import GoogLeNetOutputs

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

def one_hot_encode(labels, num_classes, type="numpy"):
    if type=="numpy":
        # Create an identity matrix of shape (num_classes, num_classes)
        identity_matrix = np.eye(num_classes)
        
        # Use the labels to index into the identity matrix
        one_hot_encoded = identity_matrix[labels]
        
        return one_hot_encoded
    elif type=="tensor":
        # Ensure labels are a tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Create a tensor of zeros with shape (len(labels), num_classes)
        one_hot_encoded = torch.zeros(labels.size(0), num_classes, dtype=torch.float)
        
        # Scatter 1s to the correct positions
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1.0)
        
        return one_hot_encoded

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

def get_covered_filenames(coverage_json_path, additional_files=None):
    """
    Loads the coverage.json file from the given path and returns a list of filenames reported,
    along with any additional files provided.
    
    :param coverage_json_path: Path to the coverage.json file
    :param additional_files: List of additional filenames to include
    :return: List of filenames covered in the report, combined with additional files
    """
    if not os.path.exists(coverage_json_path):
        raise FileNotFoundError(f"The file {coverage_json_path} does not exist.")
    
    with open(coverage_json_path, 'r') as file:
        coverage_data = json.load(file)

    # Extracting filenames from the keys of the 'files' dictionary
    filenames = list(coverage_data['files'].keys())
    
    # Add additional files if provided
    if additional_files:
        filenames.extend(additional_files)
    
    return filenames

def copy_covered_files(coverage_json_path, root_folder, additional_files=None):
    """
    Copies all the files listed in the coverage report and additional files to a specified root folder.
    
    :param coverage_json_path: Path to the coverage.json file
    :param root_folder: Path to the root folder where files should be copied
    :param additional_files: List of additional filenames to include
    """
    # Get the list of covered filenames
    filenames = get_covered_filenames(coverage_json_path, additional_files)
    
    # Delete previously extracted files
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)

    for filename in filenames:
        # Determine the source and destination paths
        src_path = os.path.abspath(filename)
        dest_path = os.path.join(root_folder, filename)
        
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
        print_to_file(f"Copied {src_path} to {dest_path}")

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

def load_and_replace_keys(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    
    # Replace all '>' with '.'
    modified_data = data.replace('>', '.')
    
    # Parse the modified string back to JSON
    return json.loads(modified_data)

def replace_in_string(data, replace_str='>', replace_with = '.'):
    # Convert the dict or JSON string to a JSON string
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = data
    
    try:
        # Replace all '>' with '.'
        modified_data_str = data_str.replace(replace_str, replace_with)
    except Exception as e:
        print_to_file(f"Replacing {replace_str} with {replace_with} in config failed with error {e}.")

    # Convert the modified string back to a dict
    return json.loads(modified_data_str)

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

def extract_output(outputs, model_type=None):
    """
    Utility function to standardize model outputs.
    
    Args:
    outputs (various): The raw outputs from the model.
    model_type (str): The type of model to adjust output extraction logic.

    Returns:
    torch.Tensor or dict: The standardized primary output tensor or dict.
    """
    
    if model_type in ['maskrcnn', 'keypointrcnn', 'fasterrcnn']:
        # For FasterRCNN, MaskRCNN, and KeypointRCNN, we expect a list of dictionaries with keys 'boxes', 'labels', 'scores', etc.
        if isinstance(outputs, list):
            if all(isinstance(elem, dict) for elem in outputs):
                # Extract the primary tensor 'boxes'
                return outputs[0]['boxes']
            else:   
                return outputs[0]
        elif all(isinstance(elem, dict) for elem in outputs):
            # Extract the primary tensor 'boxes'
            return outputs['boxes']                
        else:
            raise TypeError(f"Unexpected output format for {model_type}: {type(outputs)}")

    if isinstance(outputs, InceptionOutputs) or isinstance(outputs, GoogLeNetOutputs):
        # For InceptionV3 and GoogLeNet, return the primary output (logits) and auxiliary logits if available
        aux_logits = getattr(outputs, 'aux_logits', None)
        if aux_logits is not None:
            return outputs.logits, aux_logits
        else:
            return outputs.logits

    if isinstance(outputs, torch.Tensor):
        # If the output is already a tensor, return it directly
        return outputs
    elif isinstance(outputs, collections.OrderedDict):
        # If the output is an OrderedDict, try to return the 'out' key or the first value
        if 'out' in outputs:
            return outputs['out']
        else:
            # Return the first tensor in the OrderedDict
            return next(iter(outputs.values()))
    elif isinstance(outputs, dict):
        # If the output is a dictionary, try to return the 'out' key or the first value
        if 'out' in outputs:
            return outputs['out']
        else:
            # Return the first tensor in the dictionary
            return next(iter(outputs.values()))
    else:
        raise TypeError(f"Unsupported output type: {type(outputs)}")
        
class TqdmFile:
    def __init__(self, config=None, model_num = None):
        self.config = config
        self.model_num = model_num

    def write(self, msg):
        # Append to the file in a TQDM-compatible way
        print_to_file(msg, config=self.config, tqdm=True, model_num=self.model_num)

    def flush(self):
        pass  # No-op to conform to file interface

