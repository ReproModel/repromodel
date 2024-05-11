from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from ..decorators import enforce_types_and_ranges

class DummyPreprocessor:
    @enforce_types_and_ranges({
        'parent_input_path': {'type': str},
        'parent_output_path': {'type': str},
        'num_workers': {'type': int, 'range': (1, 32)}
    })
    def __init__(self, parent_input_path, parent_output_path, num_workers=1):
        self.parent_input_path = parent_input_path
        self.parent_output_path = parent_output_path
        self.num_workers = num_workers

    def create_paths(self, input_type):
        self.input_type = input_type
        self.data_path = Path(self.parent_input_path) / input_type
        self.output_path = Path(self.parent_output_path) / input_type
        self._create_output_dirs()

    def _create_output_dirs(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _process_file(self, file_path):
        # Load the image
        image = Image.open(file_path)
        image_array = np.array(image, dtype=np.float32)

        # Normalize the image data to 0-1
        normalized_image = image_array / 255.0

        # Save the processed image
        output_path = self.output_path / file_path.name.replace(".png", "")
        np.save(output_path, normalized_image)

    def preprocess(self):
        for input_type in ['input','target']:
            self.create_paths(input_type)

            input_paths = list(self.data_path.glob('*'))  # List all files in data_path
            log_message = f"Starting preprocessing of {len(input_paths)} files."
            print(log_message)

            if self.num_workers > 1:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    list(tqdm(executor.map(self._process_file, input_paths), total=len(input_paths)))
            else:
                for file_path in tqdm(input_paths):
                    self._process_file(file_path)

def main():
    preprocessor = DummyPreprocessor(
        parent_input_path='repromodel_core/data/dummyData',
        parent_output_path='repromodel_core/data/dummyData_preprocessed',
        num_workers=10
    )
    preprocessor.preprocess()

# if __name__ == '__main__':
#     main()