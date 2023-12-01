import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from .dataset import CustomImageFolder

import numpy as np
import argparse
import re
import os

class Loader:
    def __init__(self, root:str, args:argparse.Namespace, is_resize=False):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        self.args = args
        if is_resize:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        np.random.seed(self.args.seed)

    def load(self, dataset, dataset_path=None, num_deleted_class=0):
        """
        dataset should be "cifar100" or "dog"
        dataset_path should be specifed if dataset is "dog"
        """
        assert dataset in ["cifar100", "dog"], "dataset must be cifar100 or dog"

        if dataset == "cifar100":
            return self._load_cifar100(dataset_path, num_deleted_class)
        elif dataset == "dog":
            return self._load_dog(dataset_path, num_deleted_class)
        
    def _load_cifar100(self, dataset_path, num_deleted_class):
        """
        50000 training samples and 10000 test samples
        500 samples for each class
        """
        trainset = torchvision.datasets.CIFAR100(root=self.root,
                                                 train=True,
                                                download=True,
                                                transform=self.transform)
        testset = torchvision.datasets.CIFAR100(root=self.root,
                                                train=False,
                                                download=True,
                                                transform=self.transform)
        
        if num_deleted_class > 0:
            dataset_class = trainset.classes
            num_class = len(dataset_class)
            num_remain_class = num_class - num_deleted_class

            assert num_remain_class > 0, "num_remain_class must be positive"

            remain_class = np.random.choice(dataset_class, num_remain_class, replace=False)
            
            train_class_indices = [trainset.class_to_idx[cls] for cls in remain_class]
            train_filtered_indices = [i for i, (_, label) in enumerate(trainset) if label in train_class_indices]
            trainset = torch.utils.data.Subset(trainset, train_filtered_indices)

            test_class_indices = [testset.class_to_idx[cls] for cls in remain_class]
            test_filtered_indices = [i for i, (_, label) in enumerate(testset) if label in test_class_indices]
            testset = torch.utils.data.Subset(testset, test_filtered_indices)
            
        trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        testloader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        return trainloader, testloader
    
    def _load_dog(self, dataset_path, num_deleted_class, image_path="Images"):
        """
        20580 images
        Assume the situation that "images.tar" file provided by the Standford Dogs has been decomposed
        """
        if dataset_path is None:
            dataset_path = self.args.dataset_path
        assert dataset_path is not None, "dataset_path must be specified"

        image_dirs = os.listdir(os.path.join(self.root, dataset_path, image_path))
        pattern = re.compile(r"-([a-zA-Z_-]+)$")
        dataset_class = [re.findall(pattern, d)[0] for d in image_dirs]

        dataset = CustomImageFolder(os.path.join(self.root, dataset_path, image_path),
                                    transform=self.transform,
                                    class_list=image_dirs,
                                    custom_classes=dataset_class)
        
        if num_deleted_class > 0:
            num_class = len(dataset_class)
            num_remain_class = num_class - num_deleted_class

            assert num_remain_class > 0, "num_remain_class must be positive"

            remain_class = np.random.choice(dataset_class, num_remain_class, replace=False)

            class_indices = [dataset.class_to_idx[cls] for cls in remain_class]
            filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in class_indices]
            dataset = torch.utils.data.Subset(dataset, filtered_indices)

        train_ratio, _, test_ratio = self.args.dataset_split
        if train_ratio + test_ratio < 1.0:
            margin = 1.0 - train_ratio - test_ratio
            train_ratio += margin
        trainset, testset = torch.utils.data.random_split(dataset, [train_ratio, test_ratio])

        trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        testloader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        return trainloader, testloader




        


