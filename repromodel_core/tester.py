import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from src.getters import configure_component
from src.utils import print_to_file, load_state

SRC_DIR = "src."

# Main testing function
def test(input_data):
    # Load config
    if os.path.isfile(input_data):
        with open(input_data, 'r') as file:
            data = json.load(file)
    else:
        # Assume input is a JSON string
        data = json.loads(input_data)

    cfg = edict(data)

    # Load test dataset
    dataset_path = SRC_DIR + "datasets." + cfg.datasets[0].lower() + cfg.datasets[1:]
    test_dataset = configure_component(dataset_path, cfg.datasets, cfg.datasets_params[cfg.datasets])
    test_dataset.generate_indices(k=cfg.data_splits.k, random_seed=cfg.data_splits.random_seed)
    test_dataset.set_mode('test')
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_path)

    for model_name in cfg.models:
        print_to_file(f"Testing model: {model_name}")
        model_path = SRC_DIR + "models." + model_name[0].lower() + model_name[1:]
        model_params = cfg.models_params[model_name]
        checkpoint_path = cfg.checkpoints[model_name]
        # Load model
        model = configure_component(model_path, model_name, model_params).to(cfg.device)

        # Load model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model = load_state(model, checkpoint)

        # Configure metrics
        metrics = {}
        for metric_name in cfg.metrics:
            params = cfg.metrics_params[metric_name]
            metric_path = SRC_DIR + "metrics." + metric_name[0].lower() + metric_name[1:]
            metrics[metric_name] = configure_component(metric_path, metric_name, params)

        # Testing loop
        model.eval()
        total_metrics = {metric_name: 0.0 for metric_name in cfg.metrics}
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                inputs, targets = batch
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
            writer.add_scalar(f'Test/{model_name}/{metric_name}', value)
        
    writer.close()
    print_to_file("Testing completed and results logged to TensorBoard successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple models")
    parser.add_argument("config", type=str, help="Path to the config file or JSON string")
    args = parser.parse_args()
    test(args.config)