from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
In this code target labeled data is chosen among the unlabeled set as proposed in Mixmatch paper. (From each classes, choose n number of samples with shuffling the indices
,sample the last 500 images for each class for validation, and use whole the test batch for testing)
This code upload the pretrained encoder network and the target labeled dataset.
By using cosine similarity metric, the most similar n images are chosen among the unlabeled pool.
Chosen images are labeled with the label's of the most similar  target image.
Finally, the transfer datase is obtained by merging the target labeled dataset and the chosen images. 
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
from torch.utils.data import Dataset, DataLoader
import models
from models.resnet import *
from models.encoder import *
from models.wideresnet import Wide_ResNet
import argparse
import pickle
import numpy as np
import random

best_acc  = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--budget', choices=['4k', '2k', '1k','250'],required=True)
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the custom dataset class
class ApplyTranformDataset(Dataset):
    def __init__(self, trainset, indexes, transform=None):
        self.trainset = trainset
        self.indexes = indexes
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        image, label, original_idx = self.trainset[index]
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

Dump_File = "/home/didemgenc/Desktop/"

root=DATASET_ROOT
trainset = torchvision.datasets.CIFAR10(root, train=True, download=True,transform=None)


if args.budget =='4k':
        with open(data_path('random_4k__labeled_dataset.pkl'), 'rb') as f:
            labeled_dataset = pickle.load(f)
elif args.budget =='2k':
        with open(data_path('random_2k__labeled_dataset.pkl'), 'rb') as f:
            labeled_dataset = pickle.load(f)
elif args.budget =='1k':
        with open(data_path('random_1k__labeled_dataset.pkl'), 'rb') as f:
            labeled_dataset = pickle.load(f)
elif args.budget =='250':
        with open(data_path('random_250__labeled_dataset.pkl'), 'rb') as f:
            labeled_dataset = pickle.load(f)

indexes = list(range(len(labeled_dataset)))  # Assuming indexes are sequential
transform_applied_dataset = ApplyTranformDataset(labeled_dataset, indexes, transform=transform_test)

labeled_trainloader = DataLoader(transform_applied_dataset, batch_size=64, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root,train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

print(f"Length of loaded dataset: {len(transform_applied_dataset)}")

criterion = nn.CrossEntropyLoss()

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
        
def test_idx(epoch,target,loader):
    test_loss = 0
    correct = 0
    total = 0
    target.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(loader):
        
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        
print('\n[Test Phase] : Model setup')
net = Wide_ResNet(40, 10, 0.3, 10)
file_name = 'wide-resnet-40x10.t7'
checkpoint = torch.load(data_path('wide-resnet-40x10.t7'))
target_model = checkpoint['net']

print("Target acc on query data")
test_idx(1,target_model,labeled_trainloader)
print("Target acc on whole test data")
test(1,target_model,test_loader)

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

print("\nNow querying process is started...")
target_labels_dict = {}
target_model.eval()
target_model.training = False
correct=0
failed =0
for data, original_label, idx in labeled_trainloader:
    data = data.to(device)
    original_label = original_label.to(device)
    output = target_model(data)
    _, predicted = output.max(1)
    correct += (predicted == original_label).sum().item()
    for i in range(len(idx)):
        target_labels_dict[idx[i].item()] = predicted[i].item() # Store the image data
    #features.extend(output.tolist())

failed = len(labeled_dataset)-correct
print(f"{correct} / {failed}  acc: {correct*100/len(labeled_dataset)}: true / false classfication by target model compared to ground-truth")

# Create a new dataset with the new labels
target_labeled_set = ModifiedLabelsDataset(trainset, target_labels_dict)
target_labeled_set_for_final_dataset_concat = ModifiedLabelsDatasetforFinalDatasetConcat(trainset, target_labels_dict)
print("length of target labeled query dataset",len(target_labeled_set))
# Save the dataset as a .pkl file

if args.budget =='4k':
        with open(out_path('random_4k_target_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set, f)
        with open(out_path('random_4k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set_for_final_dataset_concat, f)
elif args.budget =='2k':
        with open(out_path('random_2k_target_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set, f)
        with open(out_path('random_2k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set_for_final_dataset_concat, f)
elif args.budget =='1k':
        with open(out_path('random_1k_target_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set, f)
        with open(out_path('random_1k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set_for_final_dataset_concat, f)
elif args.budget =='250':
        with open(out_path('random_250_target_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set, f)
        with open(out_path('random_250_target_labeled_dataset_for_final_dataset_concat.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set_for_final_dataset_concat, f)

# End time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")