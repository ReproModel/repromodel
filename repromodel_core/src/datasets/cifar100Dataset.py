import os
import pickle
import numpy as np
import torch
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset
from repromodel_core.decorators import enforce_types_and_ranges

class CIFAR100Dataset(Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    @enforce_types_and_ranges({
        'root': {'type': str},
        'train': {'type': bool},
        'transform': {'type': type(lambda x: x), 'default': None},
        'target_transform': {'type': type(lambda x: x), 'default': None},
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

        # Load the appropriate subset of the dataset
        file_path = os.path.join(self.root, self.base_folder, 'train' if self.train else 'test')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['fine_labels']  # Using fine labels for CIFAR-100

        self.data = self.data.reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC

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

