import os
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from ..decorators import enforce_types_and_ranges

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import print_to_file

class CustomPreprocessor:
    @enforce_types_and_ranges({
        'data_path': {'type': str},
        'output_path': {'type': str},
        'input_type': {'type': str, 'options': ['input', 'output', 'temp']},
        'num_workers': {'type': int, 'range': (1, 32)}  # Assuming a sensible range for number of workers
    })
    def __init__(self, data_path, output_path, input_type='input', num_workers=1):
        self.input_type = input_type
        self.output_path = Path(output_path) / input_type
        self.data_path = Path(data_path) / input_type
        self.output_path_type = Path(output_path) / input_type
        self.num_workers = num_workers
        self._create_output_dirs()

    def _create_output_dirs(self):
        self.output_path_type.mkdir(parents=True, exist_ok=True)

    def _process_file(self, file_path, output_path):
        # Define specific processing logic here
        pass

    def preprocess(self):
        input_paths = list(self.data_path.glob('*'))  # List all files in data_path
        log_message = f"Starting preprocessing of {len(input_paths)} files.\n"
        print_to_file(log_message)

        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                list(tqdm(pool.imap(self._process_file, input_paths), total=len(input_paths)))
        else:
            for file_path in tqdm(input_paths):
                self._process_file(file_path)

