import torchvision.transforms.functional as F
import torchvision.transforms as T
from .customAugmentation import CustomAugmentations
from ..decorators import enforce_types_and_ranges

class ResizePILToTensor(CustomAugmentations):
    @enforce_types_and_ranges({
        'p': {'type': float, 'range': (0.0, 1.0)},
        'height': {'type': int, 'range': (1, 10000), 'default': 224},
        'width': {'type': int, 'range': (1, 10000), 'default': 224},
    })
    def __init__(self, p, height: int, width: int):
        super().__init__(p=p)
        self.height = height
        self.width = width
        
    def get_transforms(self):
        """
        Returns a composition of transforms that resizes images and converts them to tensors.
        """
        return T.Compose([
            T.Lambda(lambda img: F.to_tensor(img)),  # Convert PIL image to tensor
            T.Resize((self.height, self.width)),
        ])