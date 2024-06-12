import albumentations as A
from albumentations.pytorch import ToTensorV2

from .customAugmentation import CustomAugmentations
from ..decorators import enforce_types_and_ranges

class ShiftScaleRotateFlip(CustomAugmentations):
    @enforce_types_and_ranges({
        'p': {'type': float, 'range': (0.0, 1.0)},
        'shift_limit': {'type': float, 'range': (-1.0, 1.0)},
        'scale_limit': {'type': float, 'range': (-1.0, 1.0)},
        'rotate_limit': {'type': int, 'range': (0, 360)}
    })
    def __init__(self, p: float, shift_limit: float, scale_limit: float, rotate_limit: int):
        super().__init__(p=p)
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.p = p
        
    def get_transforms(self):
        """
        Returns an Albumentations composition of transforms that applies random rotations and shifts.
        """
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=self.shift_limit, scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, p=self.p),
            A.HorizontalFlip(p=self.p),  # Example of another transformation with its own probability
            ToTensorV2()
        ])

# if __name__ == '__main__':
#     import cv2
#     import numpy as np
#     # Sample image: create a simple black image with a white rectangle
#     image = np.zeros((100, 100, 3), dtype=np.uint8)
#     cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)

#     # Instantiate the augmentation class
#     augmentation = ShiftScaleRotateFlip(p=0.9, shift_limit=0.2, scale_limit=0.3, rotate_limit=30)

#     # Get the transformation
#     transforms = augmentation.get_transforms()

#     # Apply transformation
#     transformed_image = transforms(image=image)['image']

#     # Show the original and transformed images
#     cv2.imshow('Original', image)
#     cv2.imshow('Transformed', transformed_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()