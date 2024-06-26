from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from ..utils import one_hot_encode
from ..decorators import enforce_types_and_ranges, tag
from typing import Any, Tuple

# In this decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["classification"], subtask=["image"], modality=["images"], submodality=["RGB"])
class CustomDataset(Dataset):
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
    @enforce_types_and_ranges({
        'input_path': {'type': str},
        'target_path': {'type': str},
        'in_channel': {'type': int, 'range': (1, 1000), 'default': 3},
        'mode': {'type': str, 'options': ['train', 'val', 'test']},
        'transforms': {'type': type(lambda x: x), 'default': None},  # Accepting function type
        'extension': {'type': str}
    })
    def __init__(self, input_path, target_path, in_channel=1, mode='train', transforms=None, extension=".nii.gz"):
        self.input_path = input_path
        self.target_path = target_path
        self.in_channel = in_channel
        self.mode = mode
        self.transforms = transforms
        self.extension = extension
        # Additional setup 
        pass

    # Required by the trainer and tester scripts
    def set_mode(self, mode: str):
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        self.mode = mode

    # Required by the trainer and tester scripts
    def set_transforms(self, transform):
        self.transform = transform

    # Required by the trainer and tester scripts
    def set_fold(self, fold: int):
        if self.indices is None:
            raise RuntimeError("Please generate indices first using generate_indices()")
        if fold >= len(self.indices):
            raise ValueError("Fold index out of range")
        self.current_fold = fold
        self.train_indices = self.indices[fold]['train']
        self.val_indices = self.indices[fold]['val']

    # Required by the trainer and tester scripts
    def generate_indices(self, k: int = 5, test_size: float = 0.2, random_seed: int = 42):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        self.indices = []
        all_indices = np.arange(len(self.data))
        for train_val_idx, test_idx in kf.split(all_indices):
            train_idx, val_idx = train_test_split(train_val_idx, test_size=test_size, random_state=random_seed)
            self.indices.append({
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': test_idx.tolist()
            })

    # Required to get the real length of every subset
    def __len__(self) -> int:
        if self.mode == 'train':
            return len(self.train_indices)
        elif self.mode == 'val':
            return len(self.val_indices)
        elif self.mode == 'test':
            return len(self.test_indices)
        return 0

    # Required to get the dataset-specific item
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.mode == 'train':
            index = self.train_indices[index]
        elif self.mode == 'val':
            index = self.val_indices[index]
        elif self.mode == 'test':
            index = self.test_indices[index]
        
        # Everything below this point is image-specific
        # Example for an image dataset:
        from PIL import Image
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            transformations = self.transform.get_transforms()
            try:
                # Try to apply torchvision transforms
                img = transformations(img)
            except TypeError:
                # Apply albumentations transforms
                img_np = np.array(img)
                transformed = transformations(image=img_np)
                img = Image.fromarray(transformed['image'])

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = one_hot_encode(labels=target, num_classes=10, type="numpy")
        return img, target
