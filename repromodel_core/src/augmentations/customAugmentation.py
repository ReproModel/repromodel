from abc import ABC, abstractmethod
from typing import Callable
from repromodel_core.src.decorators import enforce_types_and_ranges, tag
# Libraries already supported by ReproModel:
# import torchvision.transforms as T
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# In tag decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["segmentation"], subtask=["instance"], modality=["images"], submodality=["RGB"])
class CustomAugmentations(ABC):
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
    @enforce_types_and_ranges({
        'p': {'type': float, 'range': (0.0, 1.0)}  # Probability must be between 0.0 and 1.0
    })
    def __init__(self, p=1.0, **kwargs):
        """
        Initializes the CustomAugmentations class with a probability of applying the augmentation.

        Args:
            p (float): The probability of applying the augmentation, between 0.0 and 1.0.
            **kwargs: Additional keyword arguments that might be used for specific augmentations.
        """
        self.p = p
        self.kwargs = kwargs

    # Method needed by the trainer and tester scripts
    @abstractmethod
    def get_transforms(self) -> Callable:
        """
        This abstract method should be implemented by all subclasses to return the specific set of transformations.
        """
        pass