from paths import DATASET_ROOT, CIFAR_ROOT, data_path, out_path, ckpt_path
"""
In this code target labeled data is chosen among the unlabeled set as proposed in Mixmatch paper or by using active learning methods. 
(Mixmatch method: From each classes, choose n number of samples with shuffling the indices
,sample the last 500 images for each class for validation, and use whole the test batch for testing) 
"""

'''Train CIFAR10 with PyTorch.'''
from fileinput import filename
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import random
import  pickle
import os
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import detectors
import timm
from torchvision.datasets import SVHN
from torch.utils.data import Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

criterion = nn.CrossEntropyLoss()

seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize with mean and std for SVHN
                         std=[0.5, 0.5, 0.5]),
])

# Load the SVHN dataset
#trainset = SVHN(root=DATASET_ROOT, split='train', download=True, transform=None)
testset = SVHN(root=DATASET_ROOT, split='test', download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

validationset = SVHN(root=DATASET_ROOT, split='extra', download=True, transform=None)

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

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, idx):
        image, label,idx,original_label= self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, original_label 
    
    
def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    query_idxs = []
    unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        query_idxs.extend(idxs[:n_labeled_per_class])  # n_labeled_per_class images per class for training
        unlabeled_idxs.extend(idxs[n_labeled_per_class:])  # Remaining images for testing

    # Shuffle the combined lists
    np.random.shuffle(query_idxs)
    np.random.shuffle(unlabeled_idxs)

    # Select 6000 images for the test dataset
    unlabeled_idxs = unlabeled_idxs[:60000]

    return query_idxs, unlabeled_idxs


random_query_idxs, random_unlabeled_idxs = train_val_split(validationset.labels,400)

random_4k_query_dataset = CustomDataset(validationset, random_query_idxs, transform=transform)
random_4k_query_dataset_storing = CustomDataset(validationset, random_unlabeled_idxs, transform=None)
random_4k_unalabeled_dataset = CustomDataset(validationset, random_unlabeled_idxs, transform=None)

queryloader = DataLoader(random_4k_query_dataset, batch_size=128, shuffle=False, num_workers=2)

with open(out_path('problem_domain/pd_random_4k_query_dataset.pkl'), 'wb') as f:
    pickle.dump(random_4k_query_dataset_storing, f)
   
with open(out_path('problem_domain/pd_random_4k_unalabeled_dataset.pkl'), 'wb') as f:
    pickle.dump(random_4k_unalabeled_dataset, f)

print(f"Random query dataset size: {len(random_4k_query_dataset)}")
print(f"Random unlabeled dataset size: {len(random_4k_unalabeled_dataset)}")

"""
# Function to display images
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Get a batch of validation data
dataiter = iter(validationloader)
images, labels, idx = next(dataiter)

# Display a few images with their original labels
for i in range(5):  # Display 5 images
    imshow(images[i].cpu(), f'Original: {labels[i].item()}')

"""
model = timm.create_model("resnet50_svhn", pretrained=True)
model= model.to(device)

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
                         
test(1, model, testloader)

def test_idx(epoch,target,loader):
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

test(1,model,testloader)
test_idx(1,model,queryloader)

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
        return image, label, original_idx, original_label
    
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
model.eval()
model.training = False
correct=0
failed =0
for data, original_label, idx in queryloader:
    data = data.to(device)
    original_label = original_label.to(device)
    output = model(data)
    _, predicted = output.max(1)
    correct += (predicted == original_label).sum().item()
    for i in range(len(idx)):
        target_labels_dict[idx[i].item()] = predicted[i].item() # Store the image data
    #features.extend(output.tolist())

failed = len(random_4k_query_dataset)-correct
print(f"{correct} / {failed}  acc: {correct*100/len(random_4k_query_dataset)}: true / false classfication by target model compared to ground-truth")

# Create a new dataset with the new labels
target_labeled_set = ModifiedLabelsDataset(validationset, target_labels_dict)
#target_labeled_set_for_final_dataset_concat = ModifiedLabelsDatasetforFinalDatasetConcat(trainset, target_labels_dict)
 
dataset_tarnsfrom = TransformDataset(target_labeled_set, transform=transform)
dataloader_final =  DataLoader(dataset_tarnsfrom, batch_size=128, shuffle=True, num_workers=2)
print("length of target labeled query dataset",len(target_labeled_set))

#print("length of finalized dataset",len(boost_dataset))
total = len(dataset_tarnsfrom)
corr = 0
failed_image_num = 0
for _, label ,_,original_label in dataloader_final:
    corr += (label == original_label).sum().item() 

failed_image_num  = total - corr
acc = corr*100/total
print(f"{corr} / {failed_image_num} / {total} Acc: %{corr*100/total:.2f}: true / false distribution susbtitute labeled dataset!!")


with open(out_path('problem_domain/pd_svhn_4k_target_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(target_labeled_set, f)
#with open(out_path('trainset_svhn_4k_labeled_dataset_for_final_dataset_concat.pkl'), 'wb') as f:
#            pickle.dump(target_labeled_set_for_final_dataset_concat, f)
            
