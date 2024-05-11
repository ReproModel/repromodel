from pathlib import Path
from PIL import Image
from ..decorators import enforce_types_and_ranges
from .customPostprocessor import CustomPostprocessor  # Adjust the import path as necessary

class ImageToGrayscale(CustomPostprocessor):
    @enforce_types_and_ranges({
        'data_path': {'type': str},
        'output_path': {'type': str},
        'output_type': {'type': str, 'options': ['final', 'intermediate']},
        'num_workers': {'type': int, 'default': 1, 'range': (1, 100)}
    })
    def __init__(self, data_path, output_path, output_type='final', num_workers=1):
        super().__init__(data_path=data_path, output_path=output_path, output_type=output_type, num_workers=num_workers)
        self._create_output_dirs()

    def _process_data(self, file_path):
        """
        Process each image file to convert from float PNG color to int PNG grayscale.
        """
        # Open the image
        with Image.open(file_path) as img:
            # Convert the image to grayscale
            grayscale_img = img.convert('L')  # 'L' mode is for grayscale
            
            # Optionally, normalize the float image to 0-255 and convert to 'L'
            if img.mode == 'F':  # Check if image is in floating point format
                grayscale_img = Image.fromarray((255 * (grayscale_img - grayscale_img.min()) / (grayscale_img.max() - grayscale_img.min())).astype('uint8'))

            # Save the processed image to the output path
            output_file_path = Path(self.output_path_type) / file_path.name
            grayscale_img.save(output_file_path, 'PNG')

    def postprocess(self):
        super().postprocess()  # Calling the postprocess method of the base class