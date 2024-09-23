import os
import numpy as np
from PIL import Image
from torchvision.datasets import EuroSAT
from sklearn.model_selection import KFold, train_test_split
from typing import Any, Callable, List, Optional, Union, Tuple
from repromodel_core.src.decorators import enforce_types_and_ranges, tag
from ..utils import one_hot_encode
import unittest

@tag(task=["classification"], subtask=["image"], modality=["images"], submodality=["RGB"])
class EuroSATDataset(EuroSAT):
    @enforce_types_and_ranges({
        'root': {'type': str, 'default': "repromodel_core/data/eurosat"},
        'transform': {'type': Callable, 'default': None},
        'target_transform': {'type': Callable, 'default': None},
        'download': {'type': bool, 'default': False},
    })
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        self.downloaded = download
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        # Handle trainval split
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)

        # Ensure all targets are mapped correctly
        self.class_to_idx = {v: k for k, v in self.class_to_idx.items()}  # Reverse the class_to_idx mapping
        self.targets = np.array([self.class_to_idx[target] for target in self.targets])
        
        self.data = np.array(self.imgs)

        # Store the split as a string
        self.split_str = "trainval"

        # Initialize indices for cross-validation
        self.indices = None
        self.current_fold = None
        self.mode = 'train'

        # Set up all indices for the dataset
        self.all_indices = np.arange(len(self.data))

    def set_mode(self, mode: str):
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        self.mode = mode

    def set_transforms(self, transform):
        self.transform = transform.get_transforms()

    def set_fold(self, fold: int):
        if self.indices is None:
            raise RuntimeError("Please generate indices first using generate_indices()")
        if fold >= len(self.indices):
            raise ValueError("Fold index out of range")
        self.current_fold = fold
        self.train_indices = self.indices[fold]['train']
        self.val_indices = self.indices[fold]['val']
        self.test_indices = self.indices[fold]['test']

    def generate_indices(self, k: int = 5, test_size: float = 0.2, random_seed: int = 42):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        self.indices = []
        for train_val_idx, test_idx in kf.split(self.all_indices):
            train_idx, val_idx = train_test_split(train_val_idx, test_size=test_size, random_state=random_seed)
            self.indices.append({
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': test_idx.tolist()
            })

    def __len__(self) -> int:
        if self.mode == 'train' and self.train_indices:
            return len(self.train_indices)
        elif self.mode == 'val' and self.val_indices:
            return len(self.val_indices)
        elif self.mode == 'test' and self.test_indices:
            return len(self.test_indices)
        return super().__len__()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.mode == 'train' and self.train_indices:
            index = self.train_indices[index]
        elif self.mode == 'val' and self.val_indices:
            index = self.val_indices[index]
        elif self.mode == 'test' and self.test_indices:
            index = self.test_indices[index]

        img_path = self.data[index]
        target = int(img_path[1])

        img = Image.open(img_path[0]).convert("RGB")
        if self.transform is not None:
            try:
                # Try to apply torchvision transforms
                img = self.transform(img)
            except TypeError:
                # Apply albumentations transforms
                img_np = np.array(img)
                transformed = self.transform(image=img_np)
                img = Image.fromarray(transformed['image'])

        if self.target_transform is not None:
            target = self.target_transform(target)

        if isinstance(target, int):
            target = one_hot_encode(labels=target, num_classes=10, type="numpy")

        return img, target

class _TestEuroSATDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the dataset is downloaded before running tests
        EuroSATDataset(root="repromodel_core/data/eurosat", download=True)

    def test_initialization(self):
        # Test with default parameters
        dataset = EuroSATDataset(root="repromodel_core/data/eurosat", download=False)
        self.assertIsInstance(dataset, EuroSATDataset, "Dataset is not an instance of EuroSATDataset")
        self.assertEqual(dataset.split_str, "trainval", "Default split is not 'trainval'")
        self.assertFalse(dataset.downloaded, "Default download is not False")

        # Test with custom parameters
        dataset = EuroSATDataset(root="repromodel_core/data/eurosat", download=True)
        self.assertEqual(dataset.split_str, "trainval", "Custom split is not 'trainval'")
        self.assertTrue(dataset.downloaded, "Custom download is not True")
        # Check if the dataset files exist
        data_dir = os.path.join("repromodel_core/data/eurosat", "eurosat/2750")
        self.assertTrue(os.path.isdir(data_dir), "Dataset directory does not exist after downloading")
        self.assertGreater(len(os.listdir(data_dir)), 0, "Dataset directory is empty after downloading")

    def test_set_mode(self):
        dataset = EuroSATDataset(root="repromodel_core/data/eurosat")
        dataset.set_mode('val')
        self.assertEqual(dataset.mode, 'val', "Mode is not set to 'val'")

        with self.assertRaises(ValueError, msg="Setting mode to an invalid value did not raise an error"):
            dataset.set_mode('invalid')

    def test_generate_indices_and_set_fold(self):
        dataset = EuroSATDataset(root="repromodel_core/data/eurosat")
        dataset.generate_indices(k=5)
        self.assertEqual(len(dataset.indices), 5, "Number of generated folds is not 5")

        dataset.set_fold(0)
        self.assertEqual(dataset.current_fold, 0, "Current fold is not set to 0")
        self.assertIn('train', dataset.indices[0], "Train indices not found in fold 0")
        self.assertIn('val', dataset.indices[0], "Validation indices not found in fold 0")
        self.assertIn('test', dataset.indices[0], "Test indices not found in fold 0")

        with self.assertRaises(RuntimeError, msg="Setting fold without generating indices did not raise an error"):
            dataset_no_indices = EuroSATDataset(root="repromodel_core/data/eurosat")
            dataset_no_indices.set_fold(0)

        with self.assertRaises(ValueError, msg="Setting fold to an out-of-range value did not raise an error"):
            dataset.set_fold(5)

    def test_eurosat_tags(self):
        dataset = EuroSATDataset(root="repromodel_core/data/eurosat")
        self.assertEqual(dataset.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(dataset.subtask, ["image"], "Subtask tag is incorrect")
        self.assertEqual(dataset.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(dataset.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()