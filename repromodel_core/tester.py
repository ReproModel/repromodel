import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from src.getters import configure_component
from src.utils import print_to_file, load_state, get_all_ckpts, delete_command_outputs, load_and_replace_keys, replace_in_string, TqdmFile

SRC_DIR = "src."

# Main crossvalidation testing function
def test(input_data):
    # Reset the console output file
    delete_command_outputs()
    
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
            data = replace_in_string(input_data)

    cfg = edict(data)

    # Load test dataset
    dataset_path = SRC_DIR + "datasets." + cfg.datasets
    test_dataset = configure_component(dataset_path, cfg.datasets_params[cfg.datasets])
    test_dataset.generate_indices(k=cfg.data_splits.k, random_seed=cfg.data_splits.random_seed)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_path)

    # get all saved checkpoints
    checkpoints = get_all_ckpts(cfg.model_save_path, cfg.models, cfg.data_splits.k)
    for m, model_name in enumerate(cfg.models):
        # Custom file object for TQDM
        tqdm_file = TqdmFile(config=cfg, model_num = m)

        model_path = SRC_DIR + "models." + model_name 
        checkpoint_path = checkpoints[model_name]
        # Load model
        model = configure_component(model_path, cfg.models_params[model_name]).to(cfg.device)

        #add iteration over all folds
        for k in range(cfg.data_splits.k):
            print_to_file(f"Testing model {model_name} on fold {k}")
            checkpoint = torch.load(checkpoint_path[k], map_location=cfg.device)
            model = load_state(model, checkpoint)

            #configure dataloader 
            test_dataset.set_fold(k)
            test_dataset.set_mode('test')
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
                progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), file=tqdm_file)
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
                writer.add_scalar(f'CrossValTest/Fold_{k}/{model_name}/{metric_name}', value)
        
    writer.close()
    print_to_file("Cross-validation testing is completed and results are logged to TensorBoard successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple models")
    parser.add_argument("config", type=str, help="Path to the config file or JSON string")
    args = parser.parse_args()
    test(args.config)