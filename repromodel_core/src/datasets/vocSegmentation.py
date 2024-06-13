from torchvision.datasets import VOCSegmentation
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import unittest
from ..decorators import enforce_types_and_ranges, tag

@tag(task=["segmentation"], subtask=["semantic"], modality=["images"], submodality=["RGB"])
class VOCSegmentationDataset(VOCSegmentation):
    @enforce_types_and_ranges({
        'root': {'type': (str, Path), 'default': "repromodel_core/data/voc_dataset"},
        'year': {'type': str, 'default': "2012", 'options': ["2007", "2008", "2009", "2010", "2011", "2012"]},
        'image_set': {'type': str, 'default': "train", 'options': ["train", "trainval", "val", "test"]},
        'download': {'type': bool, 'default': False},
        'transform': {'type': Callable, 'default': None},
        'target_transform': {'type': Callable, 'default': None},
        'transforms': {'type': Callable, 'default': None},
        })
    def __init__(self, root, year="2012", image_set="trainval", download=False, transform=None, target_transform=None, transforms=None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        
        # Initialize indices for cross-validation
        self.download = download
        self.indices = None
        self.current_fold = None
        self.mode = 'train'

    def set_mode(self, mode: str):
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        self.mode = mode

    def set_transforms(self, transforms):
        self.transforms = transforms

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
        all_indices = np.arange(len(self.images))
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

        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            transformations = self.transforms.get_transforms()
            try:
                #torchvision transforms
                img = transformations(img)
                target = transformations(target)
            except:
                #albumentation transforms 
                img_np = np.array(img)
                target_np = np.array(target)
                transformed = transformations(image=img_np, mask=target_np)
                img, target = transformed['image'], transformed['mask']
                
        target = np.stack((target, 1-target), axis=-1)
        return img, target

class _TestVOCSegmentationDataset(unittest.TestCase):
    def test_initialization(self):
        # Test with default parameters
        dataset = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation", download=False)
        self.assertIsInstance(dataset, VOCSegmentationDataset, "Dataset is not an instance of VOCSegmentationDataset")
        self.assertEqual(dataset.year, "2012", "Default year is not 2012")
        self.assertEqual(dataset.image_set, "train", "Default image_set is not 'train'")
        self.assertFalse(dataset.download, "Default download is not False")

        # Test with custom parameters
        dataset = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation", year="2010", image_set="val", download=True)
        self.assertEqual(dataset.year, "2010", "Custom year is not 2010")
        self.assertEqual(dataset.image_set, "val", "Custom image_set is not 'val'")
        self.assertTrue(dataset.download, "Custom download is not True")

    def test_set_mode(self):
        dataset = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation")
        dataset.set_mode('val')
        self.assertEqual(dataset.mode, 'val', "Mode is not set to 'val'")

        with self.assertRaises(ValueError, msg="Setting mode to an invalid value did not raise an error"):
            dataset.set_mode('invalid')

    def test_generate_indices_and_set_fold(self):
        dataset = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation")
        dataset.generate_indices(k=5)
        self.assertEqual(len(dataset.indices), 5, "Number of generated folds is not 5")

        dataset.set_fold(0)
        self.assertEqual(dataset.current_fold, 0, "Current fold is not set to 0")
        self.assertIn('train', dataset.indices[0], "Train indices not found in fold 0")
        self.assertIn('val', dataset.indices[0], "Validation indices not found in fold 0")
        self.assertIn('test', dataset.indices[0], "Test indices not found in fold 0")

        with self.assertRaises(RuntimeError, msg="Setting fold without generating indices did not raise an error"):
            dataset_no_indices = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation")
            dataset_no_indices.set_fold(0)

        with self.assertRaises(ValueError, msg="Setting fold to an out-of-range value did not raise an error"):
            dataset.set_fold(5)
    
    def test_voc_segmentation_tags(self):
        dataset = VOCSegmentationDataset(root="repromodel_core/data/voc_segmentation")
        self.assertEqual(dataset.task, ["segmentation"], "Task tag is incorrect")
        self.assertEqual(dataset.subtask, ["semantic"], "Subtask tag is incorrect")
        self.assertEqual(dataset.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(dataset.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()