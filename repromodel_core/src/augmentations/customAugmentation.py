from abc import ABC, abstractmethod
from typing import Callable
from ..decorators import enforce_types_and_ranges

class CustomAugmentations(ABC):
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

    @abstractmethod
    def get_transforms(self) -> Callable:
        """
        This abstract method should be implemented by all subclasses to return the specific set of transformations.
        """
        pass