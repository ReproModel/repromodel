import os
import pickle
import numpy as np
import torch
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset
from repromodel_core.decorators import enforce_types_and_ranges
from torchvision.transforms import Compose

class CIFAR10Dataset(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    
    # Define train_list and test_list as class attributes
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    @enforce_types_and_ranges({
        'root': {'type': str},
        'train': {'type': bool},
        'transform': {'type': Compose, 'default': None},
        'target_transform': {'type': Compose, 'default': None},
        'download': {'type': bool}
    })
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        # Load data based on training or testing mode
        downloaded_list = self.train_list if self.train else self.test_list

        self.data = []
        self.targets = []

        # Load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        """ Check if the downloaded .tar.gz file is intact """
        filepath = os.path.join(self.root, self.filename)
        return check_integrity(filepath, self.tgz_md5)

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # Extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)