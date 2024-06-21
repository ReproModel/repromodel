import json
import importlib
from torch.utils.tensorboard import SummaryWriter
from src.utils import ensure_folder_exists
import os
import os.path
from typing import Any, List
from sklearn.model_selection import train_test_split, KFold
import numpy as np

from torch.utils.data import Dataset
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_from_lib(module_name, class_name, params):
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name, None)
    if not cls:
        raise ValueError(f"{class_name} not found in {module.__name__}")
    return cls(**params)

def get_from_module(module_path, class_name, params):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if not cls:
        raise ValueError(f"{class_name} not found in {module_path}")
    return cls(**params)

def get_optimizer(model, optimizer_name, optimizer_params):
    parts = optimizer_name.split('.')
    if len(parts) > 1:
        class_name = parts[-1]
        module_name = ".".join(parts[:-1])
    module = __import__(module_name, fromlist=[class_name])
    optimizer_class = getattr(module, class_name, None)
    if not optimizer_class:
        raise ValueError(f"Optimizer '{optimizer_name}' not found in library")

    return optimizer_class(params=model.parameters(), **optimizer_params)

def get_lr_scheduler(optimizer, scheduler_name, params):
    parts = scheduler_name.split('.')
    if len(parts) > 1:
        class_name = parts[-1]
        module_name = ".".join(parts[:-1])
    module = __import__(module_name, fromlist=[class_name])
    scheduler_class = getattr(module, class_name, None)
    
    if not scheduler_class:
        raise ValueError(
            f"LR scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler")
    return scheduler_class(optimizer, **params)

def configure_component(name, params):
    parts = name.split('.')
    name = parts[-1]
    module = ".".join(parts[:-1])
    try:
        # looks first to find the class in the source code
        return get_from_module(module, name, params)
    except:
        # if not found in the source code, look for the class in third-party libs
        return get_from_lib(module, name, params)

def configure_device_specific(component, device):
    if hasattr(component, 'to'):
        return component.to(device)
    return component

def init_tensorboard_logging(config, fold, model_num, base_dir="logs"):
    base_dir = config.tensorboard_log_path
    ensure_folder_exists(base_dir)
    subdir = f"{base_dir}/{config['models'][model_num]}_{config['datasets']}_{fold}"
    ensure_folder_exists(subdir)
    writer = SummaryWriter(subdir)
    config_path = f"{subdir}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return writer

def current_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
