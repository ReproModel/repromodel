import albumentations as A
from albumentations.pytorch import ToTensorV2

from .customAugmentation import CustomAugmentations
from ..decorators import enforce_types_and_ranges

class toTensor(CustomAugmentations):
    @enforce_types_and_ranges({
        'p': {'type': float, 'range': (0.0, 1.0), 'default': 1.0},
    })
    def __init__(self, p: float):
        super().__init__(p=p)
        self.p = p
        
    def get_transforms(self):
        """
        Returns an Albumentations composition of transforms that applies random rotations and shifts.
        """
        return A.Compose([
            ToTensorV2()
        ])
