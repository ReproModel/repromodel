import os
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from ..decorators import enforce_types_and_ranges

class DummyDataset(Dataset):
    @enforce_types_and_ranges({
        'input_path': {'type': str},
        'target_path': {'type': str},
        'in_channel': {'type': int, 'range': (1, 1000)},
        'mode': {'type': str, 'options': ['train', 'val', 'test']},
        'transforms': {'type': type(lambda x: x), 'default': None},
        'extension': {'type': str}
    })
    def __init__(self, input_path, target_path, in_channel=3, mode='train', transforms=None, extension=".npy"):
        self.input_path = input_path
        self.target_path = target_path
        self.in_channel = in_channel
        self.mode = mode
        self.transforms = transforms
        self.extension = extension

        self.input_list = self.scan_folder(self.input_path) 
        self.target_list = self.scan_folder(self.target_path)

    def set_mode(self, mode):
        """
        Set the mode of the dataset to either 'train', 'val', or 'test'.
        
        Parameters:
        - mode: str, the mode to set.
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        if mode == 'test':
            self.set_fold(0)
        self.mode = mode

    def set_transforms(self, transforms):
        self.transforms = transforms

    def scan_folder(self, dir):
        """
        Scan a folder for data files matching given extension.

        Parameters:
        - dir: The directory to scan.

        Returns:
        A list of data file paths.
        """
        abs_dir = os.path.join(os.getcwd(), dir)

        data_list = []
        assert os.path.isdir(abs_dir), '%s is not a valid directory' % abs_dir
        for root, _, fnames in sorted(os.walk(abs_dir)):
            for fname in fnames:
                if fname.startswith('.') or not any(fname.endswith(ext) for ext in self.extension):
                    continue
                data_list.append(os.path.join(root, fname))
        if len(data_list) == 0:
            raise RuntimeError(f"Found 0 files in: {abs_dir}\nSupported extension is: {self.extension}")
        return sorted(data_list)

    def generate_indices(self, k=5, test_size=0.2, random_seed=42):
        """
        Generate indices for train/test split, then apply KFold cross-validation on the training set.

        Parameters:
        - k: Number of folds.
        - test_size: Proportion of the dataset to include in the test split.
        - random_seed: Seed for randomness.
        """
        np.random.seed(random_seed)
        full_indices = np.random.permutation(len(self.input_list))
        
        # Initial train/test split
        train_val_indices, test_indices = train_test_split(full_indices, test_size=test_size, random_state=random_seed)

        kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        
        self.indices = {}
        for i, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
            self.indices[i] = {
                'train': train_idx,
                'val': val_idx,
                'test': test_indices
            }

    def set_fold(self, fold):
        """
        Set the current fold.

        Parameters:
        - fold: The fold number to set as current.
        """
        if self.indices is None:
            raise ValueError("Indices not generated. Please call generate_indices() method first.")
        if not (0 <= fold < len(self.indices)):
            raise ValueError(f"Fold number should be between 0 and {len(self.indices) - 1}")
        self.current_fold = fold

    def __len__(self):
        """
        Return the number of samples in the current fold of the dataset.
        """
        indices = self.indices[self.current_fold][self.mode]
        return len(indices)

    def __getitem__(self, idx):
        """
        Generate one sample of data.

        Parameters:
        - idx: The index of the sample to retrieve in the current fold.

        Returns:
        A tuple containing the data and its corresponding label, with the channel axis first.
        """
        actual_idx = self.indices[self.current_fold][self.mode][idx]
        input_identifier = self.input_list[actual_idx]
        target_identifier = self.target_list[actual_idx]

        data = np.load(input_identifier)
        label = np.load(target_identifier)

        if self.transforms and self.mode == 'train':
            # Get the transformation
            transformations = self.transforms.get_transforms()
            transformed = transformations(image=data, mask=label)
            data, label = transformed['image'], transformed['mask']

        # Create a two-channel label by duplicating the negative label along the channel dimension
        label = np.stack((label, 1-label), axis=-1)

        # Assuming the data and labels are stored as Height x Width x Channels
        # and we need them as Channels x Height x Width
        if self.mode in ['val','test']:
            data = np.transpose(data, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))

        return data, label

# if __name__ == "__main__":
#     import os
#     import numpy as np

#     # Define paths and parameters for testing
#     input_path = "repromodel_core/data/dummyData_preprocessed/input"
#     target_path = "repromodel_core/data/dummyData_preprocessed/target"
#     # Create an instance of your dataset
#     dataset = DummyDataset(input_path=input_path, target_path=target_path, in_channel=3, mode='train', extensions=['.npy'])

#     # Generate indices for 5-fold cross-validation
#     dataset.generate_indices(k=5)
    
#     # Set to the first fold
#     dataset.set_fold(0)
    
#     # Print basic details about the dataset
#     print("Generated indices for fold 0:")
#     print("Train Indices:", dataset.indices[0]['train'])
#     print("Validation Indices:", dataset.indices[0]['val'])
#     print("Test Indices:", dataset.indices[0]['test'])
#     print("Total training samples:", len(dataset.indices[0]['train']))
#     print("Total validation samples:", len(dataset.indices[0]['val']))
#     print("Total test samples:", len(dataset.indices[0]['test']))

#     # Print details of the first training sample
#     dataset.set_mode('train')
#     sample_data, sample_label = dataset[0]
#     print("Sample data shape:", sample_data.shape)
#     print("Sample label shape:", sample_label.shape)

#     # Optionally, toggle to validation and test modes and retrieve samples
#     dataset.set_mode('val')
#     val_data, val_label = dataset[0]
#     print("Validation data shape:", val_data.shape)
#     print("Validation label shape:", val_label.shape)

#     dataset.set_mode('test')
#     test_data, test_label = dataset[0]
#     print("Test data shape:", test_data.shape)
#     print("Test label shape:", test_label.shape)