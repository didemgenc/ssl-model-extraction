from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
This script is used to train a teacher model with a linear classifier on top of the encoder.
"""
'''Train CIFAR10 with PyTorch.'''
import time

# Start time
start_time = time.time()

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

best_acc  = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--training_method', choices=['finetune', 'layer4','frozen'],required=True)
parser.add_argument('--model', choices=['linear', 'MLP'],required=True)
parser.add_argument('--epoch', default=20, type=int)
args = parser.parse_args()


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

seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#target_labeled_dataset
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
    
class CustomDatasetUnlabeled(Dataset):
    def __init__(self, trainset):
        self.trainset = trainset

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        image, label = self.trainset[idx]
        return image, label, idx


with open(data_path('250_target_labeled_dataset.pkl'), 'rb') as f:
        transfer_dataset = pickle.load(f)

with open(data_path('random_250_test_dataset.pkl'), 'rb') as f:
        test_dataset = pickle.load(f)
    

transformed_transfer_dataset = TransformedDataset(transfer_dataset, transform=transform_train)
transfer_loader = DataLoader(transformed_transfer_dataset, batch_size=128, shuffle=True, num_workers=2)

testset = TransformedDataset(test_dataset, transform=transform_test)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

best_acc = 0.0
def subs_teacher_train(num_epochs,teacher,opt,train_loader): 
    for epoch in range(num_epochs):
        train_loss =0
        global best_acc
        correct_train =0
        total_train =0
        #true_labels=0
        teacher.train()
        for batch_idx,(inputs,targets,idx) in enumerate(train_loader):
            inputs, targets  = inputs.to(device), targets.to(device)
            #print("inputs",inputs.size())
            #print("targets",targets.size())
            #print("targets",targets)
            opt.zero_grad()
            outputs = teacher(inputs)
           
            total_loss = criterion(outputs, targets)
									
            predicted = outputs.argmax(1)
            correct_train += (predicted==targets).sum().item()
            #true_labels += (targets==o_label).sum().item()
            #print(correct_train)
            total_loss.backward()
            opt.step()
            #scheduler.step()
            train_loss += total_loss.item()
            total_train += targets.size(0)	

        valid_loss = 0.0
        correct=0
        total=0           

        teacher.eval()      
        for data, labels,_ in test_loader:
                if torch.cuda.is_available():
                    data, labels = data.to(device), labels.to(device)
                    target_v = teacher(data)
                    loss = criterion(target_v,labels)
                    predicted = target_v.argmax(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    valid_loss += loss.item()
                    valid_acc = correct / total
                 
        if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(teacher.state_dict(), ckpt_path(teach_cp))
                #print("Student saved")
            #scheduler.step()
      
        #print(f"true labels number in the {train_loader_name}: {true_labels}")
        print('(%d/%d) Substitute Model Train Loss: %.3f\t Acc: %.3f%% (%d/%d) |  Test Loss: %.3f\t Acc: %.3f%% (%d/%d)'
		                 % (epoch, num_epochs,train_loss/(batch_idx+1), 100.*correct_train/total_train, correct_train, total_train, valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
		
		
    return train_loss/(batch_idx+1), valid_loss/(batch_idx+1),100.*correct_train/total_train, 100.*correct/total


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


teach_cp = f'SubsTeach_trained_targetLabeledDataset_dataChosenwith_4k_model_{args.model}_training_{args.training_method}_test.pth'

class EncoderWithLinear(nn.Module):
    def __init__(self, encoder):
        super(EncoderWithLinear, self).__init__()
        self.encoder = encoder 
        if args.model == 'linear':
            self.linear_classifier = nn.Linear(2048, 10)
        elif args.model == 'MLP':
            #For 3 layer MLP eval. use this
            self.linear_classifier = nn.Sequential(
                nn.Linear(2048,1024),
                #nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(1024,512),
                #nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512,10)
            )        
        
    def forward(self, x):
        x = self.encoder(x, out='h')
        #x = x.view(x.size(0), -1)  # Flatten the output of the encoder
        x = self.linear_classifier(x)   
        return x

ckpt = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
encoder = EncodeProject()
encoder.load_state_dict(ckpt['state_dict'])
encoder.fc = nn.Identity()
if args.training_method == 'finetune':
    for p in encoder.parameters():
        p.requires_grad = True
elif args.training_method == 'layer4':
    for name, param in encoder.named_parameters():
        if 'layer4' not in name:
            param.requires_grad = False
elif args.training_method == 'frozen':
    for p in encoder.parameters():
        p.requires_grad = False
teacher = EncoderWithLinear(encoder)
#checkpoint = torch.load(ckpt_path(teach_cp),map_location=device)
#teacher.load_state_dict(checkpoint)
teacher = teacher.to(device)

# Print trainable layers and their names
print("Trainable layers and their names:")
for name, param in teacher.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

#print the params number of model
num_params = sum(p.numel() for p in teacher.parameters())
print(f"Number of total parameters in teacher model: {num_params}")
num_trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
print(f"Number of trainable parameters in teacher model: {num_trainable_params}")

opt_teacher = torch.optim.SGD(filter(lambda p: p.requires_grad, teacher.parameters()), lr=0.1, momentum=0.9)
#opt_teacher = torch.optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=0.001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_teacher, T_max=20)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_teacher, milestones=[40,65], gamma=0.1)
subs_teacher_train(args.epoch,teacher,opt_teacher,transfer_loader)

print("Substitute model test acc is: ")
test(1,teacher,test_loader)
end_time = time.time()
# End time
# Calculate the execution time
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
print(f"Execution time: {execution_time_seconds:.2f} seconds ({execution_time_minutes:.2f} minutes)")