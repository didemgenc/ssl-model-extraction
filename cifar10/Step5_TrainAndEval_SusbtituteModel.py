from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
This code upload the pretrained encoder network and creates  MLP or Linear evaluation model on top of it. 
Obtained model is trained with the dataset that is created by using cosine similarity between the embeddings of the target labeled dataset and the unlabeled pool dataset.
"""
import time

# Start time
start_time = time.time()
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
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset, DataLoader

import models
from models.resnet import *
from models.encoder import *
from models.wideresnet import Wide_ResNet

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
import argparse
#import optuna # hyperparam. search framework
import pickle
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import random
from torchvision.models import resnet50,resnet18, ResNet50_Weights
import imageio
# Set the print options
#torch.set_printoptions(threshold=100)
best_acc  = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epoch',default=10, type=int)
#parser.add_argument('--method', choices=['AL', 'random'],required=True)
parser.add_argument('--budget', choices=['4k', '2k','1k','250'],required=True)
parser.add_argument('--lr',default=0.1, type=float)
#parser.add_argument('--weight_decay',default=5e-4, type=float)
parser.add_argument('--whole_data_labeled_with_cossim',default=1, type=int) # if it is 0 which means all the data labeled with substitute model
parser.add_argument('--susbtitute_and_cossim',default=1, type=int) # if it is 1, dataset labeled with cossim and susbtitute model is used
parser.add_argument('--scheduler',default=1, type=int)
parser.add_argument('--threshold',choices=['4','5','9', '95','98', '8', '7','6','0'],required=True)
parser.add_argument('--substitute_threshold',choices=['0.6', '0.8','0.7'],required=True)
args = parser.parse_args()


seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)


KL_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)
criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

#whole_Data_labeled_with_cossim =1 then use this improved dataset
class ImprovedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id_list, transform = None):
        self.dataset = dataset
        self.id_list = id_list
        self.transform = transform
    
    def __len__(self):
        return len(self.id_list)
   
    def __getitem__(self, idx):
        id, label = self.id_list[idx]
        image, original_label = self.dataset[id] 
        if self.transform:
            image = self.transform(image)
        return image, label, original_label 

#unlabeled and val datasets
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
    
#apply transform to dataset   
class TransformedDataset(Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        image, label, original_idx = self.original_dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label, original_idx
    
# if in Step4 all the data is labeled by substityute model Encoder+MLP use FinalDataset 
class FinalDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_labels_dict,transform=None):
        self.original_dataset = original_dataset
        self.target_labels_dict = target_labels_dict  
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]  
        self.transform = transform

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        image, original_label = self.original_dataset[original_idx]
        if self.transform:
            image = self.transform(image)
        label = self.target_labels_dict.get(original_idx, -1)
        return image, label, original_label

#these are for substitute + cossim pseudolabeled dataset
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

class CosSimLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id_list, transform = None):
        self.dataset = dataset
        self.id_list = id_list
        self.transform = transform
    
    def __len__(self):
        return len(self.id_list)
   
    def __getitem__(self, idx):
        id, label = self.id_list[idx]
        image, original_label = self.dataset[id] 
        if self.transform:
            image = self.transform(image)
        return image, label, original_label
          
class PseudolabelByModel(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_labels_dict,transform=None):
        self.original_dataset = original_dataset
        self.target_labels_dict = target_labels_dict  
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]  
        self.transform = transform

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        image, original_label = self.original_dataset[original_idx]
        if self.transform:
            image = self.transform(image)
        label = self.target_labels_dict.get(original_idx, -1)
        return image, label, original_label
    
class FinalTransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label,original_label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label,original_label
        
#for target_labeled dataset_for_concat
class ModifiedLabelsDatasetforFinalDatasetConcat(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_labels_dict):
        self.original_dataset = original_dataset
        self.target_labels_dict = target_labels_dict  
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]  

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        image, original_label = self.original_dataset[original_idx]
        label = self.target_labels_dict.get(original_idx, -1)
        return image, label, original_label
    

if args.budget == '4k':
        with open(data_path('random_4k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
        with open(data_path('4k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
elif args.budget == '2k':
        with open(data_path('random_2k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
        with open(data_path('2k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
elif args.budget == '1k':
        with open(data_path('random_1k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
        with open(data_path('1k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
elif args.budget == '250':
        with open(data_path('random_250_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
        with open(data_path('250_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)


if args.whole_data_labeled_with_cossim == 0:
    if args.susbtitute_and_cossim == 1:
        if args.substitute_threshold == '0.6':
            if args.threshold == '98':
                if args.budget == '4k':
                    #with open(data_path('transfer_dataset/data_labeled_with_cossim_and_substitute/cossim_on_target_labeled_with_threshold/4k_0.98_confidence+cossim_labeled_1608_dataset.pkl'), 'rb') as f:
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:    
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '95':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '9':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '8':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '7':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '6':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '0':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/4k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/2k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/1k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.6substitute_threshold/250_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
        elif args.substitute_threshold == '0.7':
            if args.threshold == '98':
                if args.budget == '4k':
                    #with open(data_path('transfer_dataset/data_labeled_with_cossim_and_substitute/cossim_on_target_labeled_with_threshold/4k_0.98_confidence+cossim_labeled_1608_dataset.pkl'), 'rb') as f:
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:    
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '95':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '9':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '8':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '7':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '6':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '0':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/4k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/2k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/1k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.7substitute_threshold/250_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

        elif args.substitute_threshold == '0.8':
            if args.threshold == '98':
                if args.budget == '4k':
                    #with open(data_path('transfer_dataset/data_labeled_with_cossim_and_substitute/cossim_on_target_labeled_with_threshold/4k_0.98_confidence+cossim_labeled_1608_dataset.pkl'), 'rb') as f:
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:    
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.98_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '95':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.95_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '9':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.9_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '8':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.8_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '7':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.7_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
        
            elif args.threshold == '6':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0.6_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

            elif args.threshold == '0':
                if args.budget == '4k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/4k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '2k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/2k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '1k':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/1k_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)
                elif args.budget == '250':
                    with open(data_path('transfer_dataset/0.8substitute_threshold/250_0_confidence+cossim_labeled_dataset.pkl'), 'rb') as f:
                        train_dataset = pickle.load(f)

    elif args.susbtitute_and_cossim ==0:
        if args.threshold == '95':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.95.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
        elif args.threshold == '9':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.9.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
        elif args.threshold == '85':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.85.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
        elif args.threshold == '8':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.8.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
        elif args.threshold == '75':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.75.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)
        elif args.threshold == '7':
            with open(data_path('transfer_dataset/whole_data_pseudolabel_with_substitute/Whole_data_labeled_by_substitute_model_encoder+MLP_threshold_0.7.pkl'), 'rb') as f:
                train_dataset = pickle.load(f)


elif args.whole_data_labeled_with_cossim == 1:
    if args.threshold == '4':
        with open(data_path('transfer_dataset/4k_0.4_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '5':
        with open(data_path('transfer_dataset/4k_0.5_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '6':
        with open(data_path('transfer_dataset/4k_0.6_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '7':
        with open(data_path('transfer_dataset/4k_0.7_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '8':
        with open(data_path('transfer_dataset/4k_0.8_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '9':
        with open(data_path('transfer_dataset/4k_0.9_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 
    elif args.threshold == '95':
        with open(data_path('transfer_dataset/4k_0.95_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'rb') as f:
            train_dataset = pickle.load(f) 

#NOTE: If Step4 v1 code is executed, use this 
#transformed_target_labeled_dataset_for_final = TransformedDataset(target_labeled_dataset_for_final, transform=transform_train)
#transfomed_train_dataset = TransformedDataset(train_dataset, transform=transform_train)

testset = TransformedDataset(test_dataset, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


#NOTE: If Step4 v1 code is executed, use these
# Concatenate the datasets
#merged_dataset = torch.utils.data.ConcatDataset([transfomed_train_dataset, transformed_target_labeled_dataset_for_final])

# Create a DataLoader for the merged dataset
#merged_dataloader = DataLoader(merged_dataset, batch_size=128, shuffle=True, num_workers=2)

#print("length of finalized dataset",len(boost_dataset))

transformed_train_dataset = FinalTransformedDataset(train_dataset, transform=transform_train)
train_dataloader = DataLoader(transformed_train_dataset, batch_size=128, shuffle=True, num_workers=2)
total = len(transformed_train_dataset)
corr = 0
failed_image_num = 0
for _, label ,original_label in train_dataloader:
    corr += (label == original_label).sum().item() 

failed_image_num  = total - corr
acc = corr*100/total
print(f"{corr} / {failed_image_num} Acc: %{corr*100/total:.2f}: true / false finazalized dataset distribution")


def test(epoch,target,loader):
    test_loss = 0
    correct = 0
    total = 0
    target.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets,idx) in enumerate(loader):
        
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 


class EncoderWithLinear(nn.Module):
    def __init__(self, encoder):
        super(EncoderWithLinear, self).__init__()
        self.encoder = encoder 
        #For linear eval use this
        #self.linear_classifier = nn.Linear(2048, 10)
        
        #For 3 layer MLP eval. use this
        self.linear_classifier = nn.Sequential(
            nn.Linear(2048,1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
            #nn.ReLU(),
            #nn.Dropout(0.25),
            #nn.Linear(128,10),
        )
        
        
    def forward(self, x):
        x = self.encoder(x, out='h')
        #x = x.view(x.size(0), -1)  # Flatten the output of the encoder
        x = self.linear_classifier(x)   
        return x


best = 0.0
def train(num_epochs,loader,val_loader, model, criterion, optimizer,cp):
    for epoch in range(num_epochs):
        loss_epoch = 0
        total_train = 0
        correct_train = 0
        global best 
        model.train()
        for step, (x, y, gt_label) in enumerate(loader):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            predicted = output.argmax(1)
            correct_train += (predicted == y).sum().item()

            total_train += y.size(0)

            loss.backward()
            optimizer.step()
            

            loss_epoch += loss.item()
            
        valid_loss = 0.0
        correct_val=0
        total_val=0
        valid_acc=0
        model.eval()      
        for data, labels,idx in val_loader:
                if torch.cuda.is_available():
                    data, labels= data.to(device), labels.to(device)
                    target_v = model(data)
                    loss_val = criterion(target_v,labels)
                    predicted = target_v.argmax(1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)
                    valid_loss += loss_val.item()
        valid_acc = correct_val / total_val
             
        if valid_acc > best:
                best = valid_acc
                torch.save(model.state_dict(),cp)
                #print("Student saved")
            
        if args.scheduler == 1:
                scheduler.step()                 
                         
        print('(%d/%d) Substitute Model Train Loss: %.3f\t Acc: %.3f%% (%d/%d) |  Valid Loss: %.3f\t Acc: %.3f%% (%d/%d)'
		                 % (epoch,num_epochs,loss_epoch/(step+1), 100.*correct_train/total_train, correct_train,total_train, valid_loss/(step+1), 100.*correct_val/total_val, correct_val, total_val))
        
        #print('(%d/%d) Linear Model Train Loss: %.3f\t Acc: %.3f%% (%d/%d)'
		#                 % (epoch,num_epochs,loss_epoch/(step+1), 100.*correct_train/total_train, correct_train,total_train))

#create model
ckpt_encoder = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
encoder = EncodeProject()
encoder.load_state_dict(ckpt_encoder['state_dict'])
encoder.fc = nn.Identity()
for p in encoder.parameters():
    p.requires_grad = True
substitute_model = EncoderWithLinear(encoder)
substitute_model = substitute_model.to(device)

cp = ckpt_path("Final_substitute_model.pth")

# Print trainable layers and their names
#print("Trainable layers and their names:")
#for name, param in substitute_model.named_parameters():
#    if param.requires_grad:
#        print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

num_params = sum(p.numel() for p in substitute_model.parameters())
print(f"Number of parameters in teacher model: {num_params}")

#print(f'lr: {args.lr}, epoch: {args.epoch}, weight_decay: {args.weight_decay}, momentum: {args.momentum}')
#opt_model= torch.optim.SGD(substitute_model.parameters(), lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum) #l2 regularization;weight decay
print(f'lr: {args.lr}, epoch: {args.epoch}, budget: {args.budget} threshold: {args.threshold}')
opt_model= torch.optim.SGD(substitute_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=1e-3) #l2 regularization;weight decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=10,eta_min=1e-6)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[6,12], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.StepLR(opt_model, step_size=5, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()
train(args.epoch,train_dataloader,testloader,substitute_model,criterion,opt_model,cp)
test(1,substitute_model,testloader)

end_time = time.time()
# End time
# Calculate the execution time
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
print(f"Execution time: {execution_time_seconds:.2f} seconds ({execution_time_minutes:.2f} minutes)")