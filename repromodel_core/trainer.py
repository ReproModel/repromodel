import os
import sys
import json
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from src.getters import configure_component, get_optimizer, get_lr_scheduler, configure_device_specific, init_tensorboard_logging, load_json
from src.utils import save_model, print_to_file, delete_command_outputs, TqdmFile
from copy import deepcopy

SRC_DIR = "src."

# Main training function
def train(input_data):
    #restart command outputs file
    delete_command_outputs()
    # Load config
    if os.path.isfile(input_data):
        with open(input_data, 'r') as file:
            data = json.load(file)
    else:
        # Assume input is a JSON string
        data = json.loads(input_data)

    cfg = edict(data)

    # Set training parameters
    k_min, epoch_min, model_min = 0, 0, 0

    # load model from last checkpoint
    if cfg.load_from_checkpoint:
        metadata = load_json(cfg.metadata_path)
        cfg = edict(cfg)

        # Get model index, fold, and epoch from metadata
        model_min = cfg.models.index(metadata['model_name'])
        k_min = metadata['fold']
        epoch_min = metadata['epoch']
        print_to_file(f"Continuing training from fold # {k_min} and epoch # {epoch_min} for model {model_min} ({cfg.models[model_min]}).", config=cfg, model_num=model_min)

    # Get preprocessing, augmentation, and dataset configurations
    preprocessor_path = SRC_DIR + "preprocessing." + cfg.preprocessing[0].lower() + cfg.preprocessing[1:]
    preprocessor = configure_component(preprocessor_path, cfg.preprocessing, cfg.preprocessing_params[cfg.preprocessing])
    #preprocess the dataset
    preprocessor.preprocess()

    augmentor_path = SRC_DIR + "augmentations." + cfg.augmentations[0].lower() + cfg.augmentations[1:]
    augmentor = configure_component(augmentor_path, cfg.augmentations, cfg.augmentations_params[cfg.augmentations])
    dataset_path = SRC_DIR + "datasets." + cfg.datasets[0].lower() + cfg.datasets[1:]
    dataset = configure_component(dataset_path, cfg.datasets, cfg.datasets_params[cfg.datasets])
    dataset.generate_indices(k=cfg.data_splits.k, random_seed=cfg.data_splits.random_seed)
    dataset.set_transforms(augmentor)

    # Get metrics, model, optimizer, scheduler, loss function, and early stopper
    train_metrics, val_metrics = [], []
    
    for metric in cfg.metrics:
        params = cfg.metrics_params[metric]
        metric_path = SRC_DIR + "metrics." + metric[0].lower() + metric[1:]
        train_metrics.append(configure_component(metric_path, metric, params))
        val_metrics.append(configure_component(metric_path, metric, params))

    models = []
    for model in cfg.models:
        params = cfg.models_params[model]
        model_path = SRC_DIR + "models." + model[0].lower() + model[1:]
        models.append(configure_component(model_path, model, params))

    es_path = SRC_DIR + "early_stopping." + cfg.early_stopping[0].lower() + cfg.early_stopping[1:]

    #TODO make custom possible and refactor getters
    criterion = configure_component("torch.losses", cfg.losses, cfg.losses_params[cfg.losses])

    for m in range(model_min, len(cfg.models)):

        print_to_file("Training model " + cfg.models[m], config=cfg, model_num = m)
        # Custom file object for TQDM
        tqdm_file = TqdmFile(config=cfg, model_num = m) 

        # Training loop for each fold
        for k in range(k_min, cfg.data_splits.k):
            # Initialize TensorBoard
            writer = init_tensorboard_logging(cfg, k, m)
            model = deepcopy(models[m])
            optimizer = get_optimizer(model, "torch." + cfg.optimizers, cfg.optimizers_params[cfg.optimizers])
            lr_scheduler = get_lr_scheduler(optimizer, "torch." + cfg.lr_schedulers, cfg.lr_schedulers_params[cfg.lr_schedulers])
            early_stopper = configure_component(es_path, cfg.early_stopping, cfg.early_stopping_params[cfg.early_stopping])
            
            # Configure device specifics
            model = configure_device_specific(model, cfg.device)
            optimizer = configure_device_specific(optimizer, cfg.device)
            lr_scheduler = configure_device_specific(lr_scheduler, cfg.device)

            dataset.set_fold(k)

            train_dataset = deepcopy(dataset)
            train_dataset.set_mode('train')

            val_dataset = deepcopy(dataset)
            val_dataset.set_mode('val')

            # Prepare the DataLoader for the training dataset
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)

            # Prepare the DataLoader for the validation dataset
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False)

            best_val_loss = float('inf')
            epoch = max(0, epoch_min)
            while True:
                # Training phase
                model.train()
                total_train_loss = 0.0
                total_train_metrics = [0]*len(cfg.metrics)
                total_samples = 0

                progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=tqdm_file)
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    train_loss = criterion(outputs, labels)
                    train_loss.backward()
                    optimizer.step()

                    #caluclate and save train metrics
                    for i, metric in enumerate(train_metrics):
                        if isinstance(metric, torchmetrics.Dice):
                            labels = labels.long()
                        total_train_metrics[i] += metric(outputs, labels)

                    total_train_loss += train_loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

                    progress_bar.set_description(f"Fold {k}, Epoch {epoch} - Train Batch")
                    progress_bar.set_postfix(loss=(total_train_loss / total_samples))

                average_train_loss = total_train_loss / total_samples

                #average metrics calculation
                average_train_metrics = []
                for train_m in total_train_metrics:
                    average_train_metrics.append(train_m / total_samples)

                # Validation phase
                model.eval()
                total_val_loss = 0.0
                total_val_metrics = [0]*len(cfg.metrics)
                total_samples = 0
                progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=tqdm_file)

                with torch.no_grad():

                    for batch_idx, (inputs, labels) in progress_bar:
                        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                        outputs = model(inputs)
                        val_loss = criterion(outputs, labels)
                        total_val_loss += val_loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)

                        #caluclate and save val metrics
                        for i, metric in enumerate(val_metrics):
                            if isinstance(metric, torchmetrics.Dice):
                                labels = labels.long()
                            total_val_metrics[i] += metric(outputs, labels)
                
                #average loss calculation
                average_val_loss = total_val_loss / total_samples

                #average metrics calculation
                average_val_metrics = []
                for val_m in total_val_metrics:
                    average_val_metrics.append(val_m / total_samples)

                # Learning rate adjustment
                current_lr = optimizer.param_groups[0]['lr']
                if cfg.monitor == 'val_loss':
                    lr_scheduler.step(average_val_loss, current_lr)
                elif cfg.monitor == 'train_loss':
                    lr_scheduler.step(average_train_loss, current_lr)

                #log learning rate
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Train/Learning Rate', lr, epoch)

                #log losses
                writer.add_scalar('Train/Loss', average_train_loss, epoch)
                writer.add_scalar('Validation/Loss', average_val_loss, epoch)

                # Log metrics
                for i, metric in enumerate(cfg.metrics):
                    writer.add_scalar(f'Train/{metric}', average_train_metrics[i], epoch)
                    writer.add_scalar(f'Validation/{metric}', average_val_metrics[i], epoch)

                # Early stopping
                early_stopper.step(epoch)
                if early_stopper.should_stop:
                    print_to_file(f"Early stopping at epoch {epoch+1}", config=cfg, model_num = m)
                    writer.close()
                    break

                epoch += 1

                # Save best model
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    save_model(config=cfg, 
                               model = model, 
                               model_name=cfg.models[m], 
                               fold=k, 
                               epoch=epoch, 
                               optimizer=optimizer, 
                               lr_scheduler=lr_scheduler, 
                               early_stopping=early_stopper, 
                               train_loss=average_train_loss, 
                               val_loss=best_val_loss, 
                               is_best=True)
        print_to_file(f"Model {cfg.models[m]} training finished", config=cfg, model_num=m)

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PyTorch model')
    parser.add_argument('input_data', type=str, help='Path to the JSON request file or JSON string')
    args = parser.parse_args()
    train(args.input_data)