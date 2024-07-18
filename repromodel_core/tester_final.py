import os
import torch
from torch.utils.data import DataLoader
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from src.getters import configure_component
from src.utils import load_state, load_and_replace_keys, replace_in_string, print_to_file, TqdmFile

SRC_DIR = "src."

# Main final testing function
def test_final(input_data):
    # Load config
    # Check if input_data is a dictionary
    if isinstance(input_data, dict):
        data = replace_in_string(input_data)
    else:
        # Load config
        if os.path.isfile(input_data):
            data = load_and_replace_keys(input_data)
        else:
            # Assume input is a JSON string
            try:
                data = json.loads(input_data)
                data = replace_in_string(data)
            except json.JSONDecodeError:
                raise ValueError("Input data is neither a valid file path nor a valid JSON string")
    cfg = edict(data)

    # Load transforms
    augmentor_path = SRC_DIR + "augmentations." + cfg.augmentations
    augmentor = configure_component(augmentor_path, cfg.augmentations_params[cfg.augmentations])

    # Load test dataset
    dataset_path = SRC_DIR + "datasets." + cfg.datasets
    test_dataset = configure_component(dataset_path, cfg.datasets_params[cfg.datasets])
    test_dataset.set_transforms(augmentor)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_path)

    model_path = SRC_DIR + "models." + cfg.models[0]

    # Load model
    model = configure_component(model_path, cfg.models_params[cfg.models[0]]).to(cfg.device)

    print_to_file(f"Testing model {cfg.models} on an unseen test data")

    checkpoint = torch.load(cfg.model_checkpoint_path, map_location=cfg.device)
    model = load_state(model, checkpoint)
    print_to_file(f"Model {cfg.models} checkpoint loaded")

    #configure dataloader 
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Configure metrics
    metrics = {}
    for metric_name in cfg.metrics:
        metric_path = SRC_DIR + "metrics." + metric_name
        metrics[metric_name] = configure_component(metric_path, cfg.metrics_params[metric_name])

    # Testing loop
    model.eval()
    total_metrics = {metric_name: 0.0 for metric_name in cfg.metrics}
    num_samples = 0

    tqdm_file = TqdmFile(config=cfg, model_num = 0, mode="final_test") #model_num=0 because it is the only (final) model in a list of models
    print_to_file(f"Final testing started. \nOutput in file {cfg.tensorboard_log_path}/final_test_{cfg.models[0].split('.')[-1]}_{cfg.datasets.split('.')[-1]}" + ".txt")

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), file=tqdm_file, desc="Testing Progress")
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs)

            batch_size = inputs.size(0)
            num_samples += batch_size

            for metric_name, metric in metrics.items():
                batch_metric_value = metric(outputs, targets)
                total_metrics[metric_name] += batch_metric_value * batch_size

    # Compute average metrics
    avg_metrics = {metric_name: value / num_samples for metric_name, value in total_metrics.items()}

    # Log results to TensorBoard
    for metric_name, value in avg_metrics.items():
        writer.add_scalar(f'Final_Test/{cfg.models}/{metric_name}', value)
        
    writer.close()
    print_to_file("Final testing on unseen data is completed and results are logged to TensorBoard successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test on unseen dataset")
    parser.add_argument("config", type=str, help="Path to the config file or JSON string")
    args = parser.parse_args()
    test_final(args.config)