from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
In this code target labeled data is chosen among the unlabeled set as proposed in Mixmatch paper or by using active learning methods. 
(Mixmatch method: From each classes, choose n number of samples with shuffling the indices
,sample the last 500 images for each class for validation, and use whole the test batch for testing) 
"""

'''Train CIFAR10 with PyTorch.'''
from fileinput import filename
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
import  pickle
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
with open(data_path('like_mixmatch_labeled_unlabeled_set_split/mixmatch_like_4k_target_labeled_set.pkl'), 'rb') as f:
    labeled_dataset = pickle.load(f)

with open(data_path('like_mixmatch_labeled_unlabeled_set_split/mixmatch_like_train_unlabeled_dataset.pkl'), 'rb') as f:
    unlabeled_dataset = pickle.load(f)

with open(data_path('like_mixmatch_labeled_unlabeled_set_split/mixmatch_like_val_dataset.pkl'), 'rb') as f:
    validation_dataset = pickle.load(f)

testset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True,transform=transform_test)
whole_test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

labeled_trainloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=0)
unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=0)

#creating 4k target labeled dataset, 5k validaton set among trainset and 41k unlabeled dataset
"""

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, trainset, indexes, transform=None):
        self.trainset = trainset
        self.indexes = indexes
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        image, label = self.trainset[index]
        if self.transform:
            image = self.transform(image)
        return image, label, index
    


# Define the custom dataset class
class CustomDatasetUnlabeled(Dataset):
    def __init__(self, trainset):
        self.trainset = trainset

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        image, label = self.trainset[idx]
        return image, label, idx
    

    
def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    test_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])  # n_labeled_per_class images per class for training
        test_idxs.extend(idxs[n_labeled_per_class:])  # Remaining images for testing

    # Shuffle the combined lists
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(test_idxs)

    # Select 6000 images for the test dataset
    test_idxs = test_idxs[:6000]

    return train_labeled_idxs, test_idxs


root=DATASET_ROOT
testset = torchvision.datasets.CIFAR10(root, train=False, download=True)
trainset = torchvision.datasets.CIFAR10(root, train=True, download=True)

#random_labeled_idxs, random_test_idxs = train_val_split(testset.targets, 200)

#random_2k__labeled_dataset = CustomDataset(testset, random_labeled_idxs, transform=None)
#random_2k_test_dataset = CustomDataset(testset, random_test_idxs, transform=None)
unlabeled_4k_dataset = CustomDatasetUnlabeled(trainset)

#with open(out_path('random_2k__labeled_dataset.pkl'), 'wb') as f:
#    pickle.dump(random_2k__labeled_dataset, f)
   
#with open(out_path('random_2k_test_dataset.pkl'), 'wb') as f:
#    pickle.dump(random_2k_test_dataset, f)

with open(out_path('unlabeled_4k_dataset.pkl'), 'wb') as f:
    pickle.dump(unlabeled_4k_dataset, f)

#print(f"Random labeled dataset size: {len(random_2k__labeled_dataset)}")
#print(f"Random test dataset size: {len(random_2k_test_dataset)}")




