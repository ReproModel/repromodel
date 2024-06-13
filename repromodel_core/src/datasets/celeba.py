import os
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import KFold, train_test_split
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets import CelebA
from typing import Any, Callable, List, Optional, Union, Tuple
from ..decorators import enforce_types_and_ranges, tag
import unittest
import pandas as pd

@tag(task=["classification"], subtask=["attribute"], modality=["images"], submodality=["RGB"])
class CelebADataset(CelebA):
    @enforce_types_and_ranges({
        'root': {'type': str, 'default': "repromodel_core/data/celeba"},
        'split': {'type': str, 'default': "train", 'options': ["train", "valid", "test", "trainval", "all"]},
        'target_type': {'type': (str, list), 'default': "attr", 'options': ["attr", "identity", "bbox", "landmarks"]},
        'transform': {'type': Callable, 'default': None},
        'target_transform': {'type': Callable, 'default': None},
        'download': {'type': bool, 'default': False},
    })
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, 
                 download: bool = False) -> None:
        # split = "train" passed per default because it overridden later in the code
        super().__init__(root, split="train", target_type=target_type, transform=transform, 
                         target_transform=target_transform, download=download)

        # Initialize indices for cross-validation
        self.indices = None
        self.current_fold = None
        self.mode = 'train'
        self.split = split
        
        # Load split information
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "trainval": [0, 1],
            "all": None,
        }
        self.split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "trainval", "all"))]

        # Load necessary data
        self.splits = pd.read_csv(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"), sep='\s+', header=None, index_col=0)
        self.identity = pd.read_csv(os.path.join(self.root, self.base_folder, "identity_CelebA.txt"), sep='\s+', header=None, index_col=0)
        self.bbox = pd.read_csv(os.path.join(self.root, self.base_folder, "list_bbox_celeba.txt"), sep='\s+', header=1, index_col=0)
        self.landmarks_align = pd.read_csv(os.path.join(self.root, self.base_folder, "list_landmarks_align_celeba.txt"), sep='\s+', header=1)
        self.attr = pd.read_csv(os.path.join(self.root, self.base_folder, "list_attr_celeba.txt"), sep='\s+', header=1)

        if self.split_ == [0, 1]:
            train_mask = (self.splits[1] == 0)
            valid_mask = (self.splits[1] == 1)
            self.mask = train_mask | valid_mask
        else:
            self.mask = slice(None) if self.split_ is None else (self.splits[1] == self.split_)

        self.filename = self.splits[self.mask].index.values
        self.identity = torch.as_tensor(self.identity[self.mask].values)
        self.bbox = torch.as_tensor(self.bbox[self.mask].values)
        self.landmarks_align = torch.as_tensor(self.landmarks_align[self.mask].values)
        self.attr = torch.as_tensor(self.attr[self.mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}

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
        self.test_indices = self.indices[fold]['test']

    def generate_indices(self, k: int = 5, test_size: float = 0.2, random_seed: int = 42):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        self.indices = []
        all_indices = np.arange(len(self.filename))
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

        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f"Target type '{t}' is not recognized.")

        if self.transform is not None:
            transformations = self.transform.get_transforms()
            try:
                # Try to apply torchvision transforms
                X = transformations(X)
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X)  # Ensure X is a tensor
            except TypeError:
                # Apply albumentations transforms
                X_np = np.array(X)
                transformed = transformations(image=X_np)
                X = transformed['image']
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X)  # Ensure X is a tensor

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        return X, target

class _TestCelebADataset(unittest.TestCase):
    def test_initialization(self):
        # Test with default parameters
        dataset = CelebADataset(root="repromodel_core/data/celeba", download=False)
        self.assertIsInstance(dataset, CelebADataset, "Dataset is not an instance of CelebADataset")
        self.assertEqual(dataset.split, "train", "Default split is not 'train'")
        self.assertEqual(dataset.target_type, ["attr"], "Default target_type is not 'attr'")
        self.assertTrue(dataset.download, "Default download is not False")

        # Test with custom parameters
        dataset = CelebADataset(root="repromodel_core/data/celeba", split="valid", target_type="identity", download=True)
        self.assertEqual(dataset.split, "valid", "Custom split is not 'valid'")
        self.assertEqual(dataset.target_type, ["identity"], "Custom target_type is not 'identity'")
        self.assertTrue(dataset.download, "Custom download is not True")

        # Test with trainval split
        dataset = CelebADataset(root="repromodel_core/data/celeba", split="trainval", download=False)
        self.assertEqual(dataset.split, "trainval", "Custom split is not 'trainval'")
        self.assertGreater(len(dataset.filename), 0, "trainval split should have more than 0 samples")

    def test_set_mode(self):
        dataset = CelebADataset(root="repromodel_core/data/celeba")
        dataset.set_mode('val')
        self.assertEqual(dataset.mode, 'val', "Mode is not set to 'val'")

        with self.assertRaises(ValueError, msg="Setting mode to an invalid value did not raise an error"):
            dataset.set_mode('invalid')

    def test_generate_indices_and_set_fold(self):
        dataset = CelebADataset(root="repromodel_core/data/celeba")
        dataset.generate_indices(k=5)
        self.assertEqual(len(dataset.indices), 5, "Number of generated folds is not 5")

        dataset.set_fold(0)
        self.assertEqual(dataset.current_fold, 0, "Current fold is not set to 0")
        self.assertIn('train', dataset.indices[0], "Train indices not found in fold 0")
        self.assertIn('val', dataset.indices[0], "Validation indices not found in fold 0")
        self.assertIn('test', dataset.indices[0], "Test indices not found in fold 0")

        with self.assertRaises(RuntimeError, msg="Setting fold without generating indices did not raise an error"):
            dataset_no_indices = CelebADataset(root="repromodel_core/data/celeba")
            dataset_no_indices.set_fold(0)

        with self.assertRaises(ValueError, msg="Setting fold to an out-of-range value did not raise an error"):
            dataset.set_fold(5)
    
    def test_celeb_a_tags(self):
        dataset = CelebADataset(root="repromodel_core/data/celeba")
        self.assertEqual(dataset.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(dataset.subtask, ["attribute"], "Subtask tag is incorrect")
        self.assertEqual(dataset.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(dataset.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()