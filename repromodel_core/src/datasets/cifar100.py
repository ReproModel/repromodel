import os
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100
from sklearn.model_selection import KFold, train_test_split
from typing import Any, Callable, List, Optional, Union, Tuple
from ..decorators import enforce_types_and_ranges, tag
from ..utils import one_hot_encode
import unittest

@tag(task=["classification"], subtask=["image"], modality=["images"], submodality=["RGB"])
class CIFAR100Dataset(CIFAR100):
    @enforce_types_and_ranges({
        'root': {'type': str, 'default': "repromodel_core/data/cifar100"},
        'train': {'type': bool, 'default': True},
        'transform': {'type': Callable, 'default': None},
        'target_transform': {'type': Callable, 'default': None},
        'download': {'type': bool, 'default': False},
    })
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        self.downloaded = download
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Initialize indices for cross-validation
        self.indices = None
        self.current_fold = None
        self.mode = 'train'
        self.train_data = self.data
        self.train_targets = self.targets

        if not train:
            self.test_data = self.data
            self.test_targets = self.targets

    def set_mode(self, mode: str):
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        self.mode = mode

    def set_transforms(self, transform):
        self.transform = transform

    def set_fold(self, fold: int):
        if self.indices is None:
            raise RuntimeError("Please generate indices first using generate_indices()")
        if fold >= len(self.indices):
            raise ValueError("Fold index out of range")
        self.current_fold = fold
        self.train_indices = self.indices[fold]['train']
        self.val_indices = self.indices[fold]['val']

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

    def __len__(self) -> int:
        if self.mode == 'train':
            return len(self.train_indices)
        elif self.mode == 'val':
            return len(self.val_indices)
        elif self.mode == 'test':
            return len(self.test_indices)
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.mode == 'train':
            index = self.train_indices[index]
        elif self.mode == 'val':
            index = self.val_indices[index]
        elif self.mode == 'test':
            index = self.test_indices[index]

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

        target = one_hot_encode(labels=target, num_classes=100, type="numpy")
        return img, target
    
class _TestCIFAR100Dataset(unittest.TestCase):
    def test_initialization(self):
        # Test with default parameters
        dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100", download=False)
        self.assertIsInstance(dataset, CIFAR100Dataset, "Dataset is not an instance of CIFAR100Dataset")
        self.assertTrue(dataset.train, "Default train is not True")
        self.assertFalse(dataset.downloaded, "Default download is not False")

        # Test with custom parameters
        dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100", train=False, download=True)
        self.assertFalse(dataset.train, "Custom train is not False")
        self.assertTrue(dataset.downloaded, "Custom download is not True")
        # Check if the dataset files exist
        data_dir = os.path.join("repromodel_core/data/cifar100", "cifar-100-python")
        self.assertTrue(os.path.isdir(data_dir), "Dataset directory does not exist after downloading")
        self.assertGreater(len(os.listdir(data_dir)), 0, "Dataset directory is empty after downloading")

    def test_set_mode(self):
        dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100")
        dataset.set_mode('val')
        self.assertEqual(dataset.mode, 'val', "Mode is not set to 'val'")

        with self.assertRaises(ValueError, msg="Setting mode to an invalid value did not raise an error"):
            dataset.set_mode('invalid')

    def test_generate_indices_and_set_fold(self):
        dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100")
        dataset.generate_indices(k=5)
        self.assertEqual(len(dataset.indices), 5, "Number of generated folds is not 5")

        dataset.set_fold(0)
        self.assertEqual(dataset.current_fold, 0, "Current fold is not set to 0")
        self.assertIn('train', dataset.indices[0], "Train indices not found in fold 0")
        self.assertIn('val', dataset.indices[0], "Validation indices not found in fold 0")
        self.assertIn('test', dataset.indices[0], "Test indices not found in fold 0")

        with self.assertRaises(RuntimeError, msg="Setting fold without generating indices did not raise an error"):
            dataset_no_indices = CIFAR100Dataset(root="repromodel_core/data/cifar100")
            dataset_no_indices.set_fold(0)

        with self.assertRaises(ValueError, msg="Setting fold to an out-of-range value did not raise an error"):
            dataset.set_fold(5)

    def test_cifar100_tags(self):
        dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100")
        self.assertEqual(dataset.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(dataset.subtask, ["image"], "Subtask tag is incorrect")
        self.assertEqual(dataset.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(dataset.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    #for the first run
    #dataset = CIFAR100Dataset(root="repromodel_core/data/cifar100", download=True)
    unittest.main()