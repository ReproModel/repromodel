import os
import torch
from torch.utils.data import DataLoader
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from src.getters import configure_component
from src.utils import load_state, load_and_replace_keys, replace_in_string

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

    # Load test dataset
    dataset_path = SRC_DIR + "datasets." + cfg.datasets
    test_dataset = configure_component(dataset_path, cfg.datasets_params[cfg.datasets])

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_path)

    model_path = SRC_DIR + "models." + cfg.model_name 

    # Load model
    model = configure_component(model_path, cfg.models_params[cfg.model_name ]).to(cfg.device)

    print(f"Testing model {cfg.model_name} on an unseen test data")

    checkpoint = torch.load(cfg.model_checkpoint_path, map_location=cfg.device)
    model = load_state(model, checkpoint)
    print(f"Model {cfg.model_name} checkpoint loaded")

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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
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
        writer.add_scalar(f'Final_Test/{cfg.model_name}/{metric_name}', value)
        
    writer.close()
    print("Final testing on unseen data is completed and results are logged to TensorBoard successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test on unseen dataset")
    parser.add_argument("config", type=str, help="Path to the config file or JSON string")
    args = parser.parse_args()
    test_final(args.config)