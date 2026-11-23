from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
This code is used to create the baseline supervised model which is Resnet50 trained on directly 4k transfer dataset which is labeled by target model.
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
from torch.utils.data import Dataset, DataLoader
import models
from models.resnet import *
from models.encoder import *
from torchvision.models import resnet50, ResNet50_Weights
from models.wideresnet import Wide_ResNet
from torchvision.models import resnet50
from torchvision.datasets import SVHN
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import argparse
#import optuna # hyperparam. search framework
import pickle
from tqdm import tqdm
import random
import imageio


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")


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


class ModifiedLabelsDataset(torch.utils.data.Dataset):
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
        return image, label, original_idx

    
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, idx):
        image, label,original_id = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx 
    
seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

criterion = nn.CrossEntropyLoss()
root=DATASET_ROOT

with open(data_path('250_target_labeled_dataset.pkl'), 'rb') as f:
        target_labeled_dataset = pickle.load(f)


trainset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True,transform=None)

transformed_target_labeled_dataset = TransformDataset(target_labeled_dataset, transform=transform_train)
target_labeled_loader = DataLoader(transformed_target_labeled_dataset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root,train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

def test(epoch,target,loader):
    test_loss = 0
    correct = 0
    total = 0
    target.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
        
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 


def train(num_epochs,loader,model, criterion, optimizer):
    for epoch in range(num_epochs):
        loss_epoch = 0
        total_train = 0
        correct_train = 0
        global best 
        model.train()
        for step, (x, y, id) in enumerate(loader):
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
            #scheduler.step()

            loss_epoch += loss.item()

        valid_loss = 0.0
        correct_val=0
        total_val=0
        model.eval()      
        for data, labels in test_loader:
                if torch.cuda.is_available():
                    data, labels= data.to(device), labels.to(device)
                    target_v = model(data)
                    loss_val = criterion(target_v,labels)
                    predicted = target_v.argmax(1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)
                    valid_loss += loss_val.item()
        valid_acc = correct_val / total_val
                          
                         
        print('(%d/%d) Model Train Loss: %.2f\t Acc: %.2f%% (%d/%d) |  Valid Loss: %.2f\t Acc: %.2f%% (%d/%d)'
		                 % (epoch,num_epochs,loss_epoch/(step+1), 100.*correct_train/total_train, correct_train,total_train, valid_loss/(step+1), 100.*correct_val/total_val, correct_val, total_val))


model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
for params in model.parameters():
                    params.requires_grad_ = True                
num_ftrs_model= model.fc.in_features                
model.fc = nn.Linear(num_ftrs_model,10)   
model.to(device) 

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

train(40,target_labeled_loader,model,criterion,optimizer)
test(1,model,test_loader)
