import json
import torch
import torch.nn as nn
import importlib
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from src.utils import ensure_folder_exists

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_from_torch(module, class_name, params):
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
    if optimizer_name.split('.')[0] == 'torch':
        optimizer_name = optimizer_name.split('.')[1]

    optimizer_class = getattr(torch.optim, optimizer_name, None)
    if not optimizer_class:
        raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")

    return optimizer_class(params=model.parameters(), **optimizer_params)

def get_lr_scheduler(optimizer, scheduler_name, params):
    if scheduler_name.split('.')[0] == 'torch':
        scheduler_name = scheduler_name.split('.')[1]
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name, None)
    
    if not scheduler_class:
        raise ValueError(
            f"LR scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler")
    return scheduler_class(optimizer, **params)

def configure_component(type, name, params):
    if "torch" in type:
        parts = name.split('.')
        if len(parts) > 1:
            name = parts[-1]
        module = nn if 'loss' in type \
                 else torchmetrics
        return get_from_torch(module, name, params)
    else:
        return get_from_module(type, name, params)

def configure_device_specific(component, device):
    if hasattr(component, 'to'):
        return component.to(device)
    return component

def init_tensorboard_logging(config, fold, model_num, base_dir="logs"):
    base_dir = config.tensorboard_log_path
    ensure_folder_exists(base_dir)
    subdir = f"{base_dir}/{config['model'][model_num]}_{config['dataset']}_{fold}"
    ensure_folder_exists(subdir)
    writer = SummaryWriter(subdir)
    config_path = f"{subdir}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return writer

def current_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
