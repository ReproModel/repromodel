import os
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from repromodel_core.src.decorators import enforce_types_and_ranges, tag
from utils import print_to_file
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# In tag decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["classification"], subtask=["multi-class"], modality=["medical"], submodality=["CT"])
class CustomPostprocessor:
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
    @enforce_types_and_ranges({
        'data_path': {'type': str},
        'output_path': {'type': str},
        'output_type': {'type': str, 'options': ['final', 'intermediate', 'logs']},
        'num_workers': {'type': int, 'range': (1, 32)}
    })
    def __init__(self, data_path, output_path, output_type='final', num_workers=1):
        self.output_type = output_type
        self.output_path = Path(output_path) / output_type
        self.data_path = Path(data_path)
        self.output_path_type = Path(output_path) / output_type
        self.num_workers = num_workers
        self._create_output_dirs()

    def _create_output_dirs(self):
        self.output_path_type.mkdir(parents=True, exist_ok=True)

    def _process_data(self, file_path):
        # Implement your specific post-processing logic here
        # Example: Analyze data, modify, summarize, or prepare for visualization
        pass

    def postprocess(self):
        processed_paths = list(self.data_path.glob('*'))  # List all processed files
        log_message = f"Starting post-processing of {len(processed_paths)} files.\n"
        print_to_file(log_message)

        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                list(tqdm(pool.imap(self._process_data, processed_paths), total=len(processed_paths)))
        else:
            for file_path in tqdm(processed_paths):
                self._process_data(file_path)