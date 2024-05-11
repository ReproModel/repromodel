import os
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
from ..decorators import enforce_types_and_ranges

class CustomDataset(Dataset):
    @enforce_types_and_ranges({
        'input_path': {'type': str},
        'target_path': {'type': str},
        'in_channel': {'type': int, 'range': (1, 1000)},
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

    def set_transforms(self, transforms):
        self.transforms = transforms

    def scan_folder(self, dir):
        """
        Scan a folder for data.

        Parameters:
        - dir: The directory to scan for data.

        Returns:
        A list of data file paths.
        """
        data_list = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if any(part for part in fname.split(os.sep) if part.startswith('.')):
                    continue
                if any(fname.endswith(extension) for extension in (self.image_ext + self.nifti_ext)):
                    data_list.append(os.path.join(root, fname))
        if len(data_list) == 0:
            raise(RuntimeError("Found 0 files in: " + dir + "\n"
                               "Supported extensions are: " +
                               ",".join(self.image_ext + self.nifti_ext)))
        return sorted(data_list)


    def load_test_indices(self, indices, fold):
        """
        Load test indices.

        Parameters:
        - indices: The indices to load.
        - fold: The fold number to load.
        """

        self.indices = {fold: {'test': indices}}
        self.current_fold = fold
    
    def generate_indices(self, k=5, random_seed=42):
        """
        Generate indices for KFold cross-validation.

        Parameters:
        - k: Number of folds for KFold cross-validation.
        - random_seed: Random seed for reproducibility in KFold.
        """

        train_idx, test_idx = self.train_test_split_indices(len(self.input_list), random_seed=random_seed)

        train_list =  [self.input_list[i] for i in train_idx]
        mapping = {i: self.input_list.index(train_list[i]) for i in range(len(train_list))}

        kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)

        self.indices = {i: {'train': [mapping[j] for j in train_idx],
                            'val': [mapping[j] for j in val_idx],
                            'test': test_idx} for i, (train_idx, val_idx) in enumerate(kfold.split(train_list))}
        

    def train_test_split_indices(self, data_length, test_size=0.2, random_seed=42):
        """
        Split data indices based on the length of the data into training and validation sets.
    
        Args:
            data_length (int): The total number of data points.
            test_size (float): The proportion of data to include in the validation set.
            random_seed (int): Random seed for reproducibility.
    
        Returns:
            dict: A dictionary containing "train" and "test" lists of indices.
        """
        # Generate a list of data indices based on the data length
        data_indices = list(range(data_length))
        
        # Set the random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Shuffle the data indices randomly
        np.random.shuffle(data_indices)
        
        # Calculate the split point
        split_point = int(data_length * (1 - test_size))
        
        # Split the data indices into training and validation sets
        train_indices = data_indices[:split_point]
        test_indices = data_indices[split_point:]
        
        return train_indices, test_indices


    def set_fold(self, fold):
        """
        Set the current fold.

        Parameters:
        - fold: The fold number to set as current.
        """
        if self.indices is None:
            raise ValueError("Indices not generated. Please call generate_indices() method first.")
        if fold < 0 or fold >= len(self.indices):
            raise ValueError(f"Fold number should be between 0 and {len(self.indices) - 1}")
        self.current_fold = fold

    def set_mode(self, mode: str):
        """Sets the mode of the dataset."""
        self.mode = mode

    def __len__(self):
        """
        Return the number of samples in the current fold of the dataset.
        """
        if self.current_fold is None:
            raise ValueError("Fold is not set. Please call set_fold() method first.")
        
        indices = self.indices[self.current_fold][self.mode]
        return len(indices)


    def __getitem__(self, idx):
        """
        Generate one sample of data.

        Parameters:
        - idx: The index of the sample to retrieve in the current fold.

        Returns:
        A tuple containing the data and its corresponding label.
        """

        if self.current_fold is None:
            raise ValueError("Fold is not set. Please call set_fold() method first.")

        input_identifier = self.input_list[idx]
        target_identifier = self.target_list[idx]

        data = self.reader(input_identifier)
        label = self.reader(target_identifier)

        if self.transforms:
            # transformed = self.transforms[self.mode]({'image': data, 'target': label})
            # data, label = transformed['image'], transformed['target']
            pass

        return data, label
