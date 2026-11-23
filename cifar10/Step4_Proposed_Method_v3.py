from paths import DATASET_ROOT, data_path, out_path, ckpt_path
""" Bu kodda sadece encoder+MLP modeli kullanılarak pseudolabellanan samplerdan yüksek threshold  sahip olanlar kullanılmıştır. 
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
from torch.utils.data import Dataset,DataLoader

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
import pickle
from tqdm import tqdm
import random


best_acc  = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--budget', choices=['4k', '2k','1k','250'],required=True)
parser.add_argument('--threshold', default =0.95, type=float)
#parser.add_argument('--model', choices=['linear', 'MLP'],required=True)
#parser.add_argument('--epoch', default=20, type=int)
args = parser.parse_args()

seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

# Define the custom dataset class
class CustomDatasetUnlabeled(Dataset):
    def __init__(self, trainset):
        self.trainset = trainset

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        image, label = self.trainset[idx]
        return image, label, idx
    

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
    
seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.budget == '4k':
    with open(data_path('4k_target_labeled_dataset.pkl'), 'rb') as f:
            target_labeled_dataset = pickle.load(f)
    with open(data_path('4k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
    with open(data_path('random_4k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)  
    with open(data_path('unlabeled_4k_dataset.pkl'), 'rb') as f:
            unlabeled_dataset = pickle.load(f)     
elif args.budget == '2k':
    with open(data_path('2k_target_labeled_dataset.pkl'), 'rb') as f:
            target_labeled_dataset = pickle.load(f)
    with open(data_path('2k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
    with open(data_path('random_2k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)  
    with open(data_path('unlabeled_2k_dataset.pkl'), 'rb') as f:
            unlabeled_dataset = pickle.load(f)    
elif args.budget == '1k':
    with open(data_path('1k_target_labeled_dataset.pkl'), 'rb') as f:
            target_labeled_dataset = pickle.load(f)
    with open(data_path('1k_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
    with open(data_path('random_1k_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)  
    with open(data_path('unlabeled_1k_dataset.pkl'), 'rb') as f:
            unlabeled_dataset = pickle.load(f)  
elif args.budget == '250':
    with open(data_path('250_target_labeled_dataset.pkl'), 'rb') as f:
            target_labeled_dataset = pickle.load(f)
    with open(data_path('250_target_labeled_dataset_for_final_dataset_concat.pkl'), 'rb') as f:
            target_labeled_dataset_for_final = pickle.load(f)
    with open(data_path('random_250_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)  
    with open(data_path('unlabeled_250_dataset.pkl'), 'rb') as f:
            unlabeled_dataset = pickle.load(f)  

print(len(test_dataset))

trainset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True,transform=None)

#transformed_target_labeled_dataset = TransformedDataset(target_labeled_dataset, transform=transform_train)
#target_labeled_loader = DataLoader(transformed_target_labeled_dataset, batch_size=128, shuffle=False, num_workers=2)

transformed_target_labeled_dataset = TransformedDataset(target_labeled_dataset, transform=transform_test)
target_labeled_loader_untransformed = DataLoader(transformed_target_labeled_dataset, batch_size=128, shuffle=False, num_workers=2)

transformed_target_labeled_dataset_for_final = TransformedDataset(target_labeled_dataset_for_final, transform=transform_train)
#target_labeled_loader_untransformed = DataLoader(target_labeled_dataset_for_final, batch_size=128, shuffle=False, num_workers=2)

transformed_unlabeled_dataset = TransformedDataset(unlabeled_dataset, transform=transform_test)
unlabeled_loader = DataLoader(transformed_unlabeled_dataset, batch_size=128, shuffle=False, num_workers=2)

testset = TransformedDataset(test_dataset, transform=transform_test)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def train(num_epochs,model,opt,train_loader): 
    for epoch in range(num_epochs):
        train_loss =0
        global best_acc
        correct_train =0
        total_train =0
        #true_labels=0
        model.train()
        for batch_idx,(inputs,targets,original_label) in enumerate(train_loader):
            inputs, targets  = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
           
            total_loss = criterion(outputs, targets)
									
            predicted = outputs.argmax(1)
            correct_train += (predicted==targets).sum().item()
            #true_labels += (targets==o_label).sum().item()
            #print(correct_train)
            total_loss.backward()
            opt.step()
            scheduler.step()
            train_loss += total_loss.item()
            total_train += targets.size(0)	

            valid_loss = 0.0
            correct=0
            total=0           

            model.eval()      
            for data, labels,_ in val_loader:
                if torch.cuda.is_available():
                    data, labels = data.to(device), labels.to(device)
                    target_v = model(data)
                    loss = criterion(target_v,labels)
                    predicted = target_v.argmax(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    valid_loss += loss.item()
                    valid_acc = correct / total
                 
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(),cp)
                #print("Student saved")
      
        #print(f"true labels number in the {train_loader_name}: {true_labels}")
        print('(%d/%d) Final Model Train Loss: %.3f\t Acc: %.3f%% (%d/%d) |  Valid Loss: %.3f\t Acc: %.3f%% (%d/%d)'
		                 % (epoch, num_epochs,train_loss/(batch_idx+1), 100.*correct_train/total_train, correct_train, total_train, valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
		
		
    return train_loss/(batch_idx+1), valid_loss/(batch_idx+1),100.*correct_train/total_train, 100.*correct/total
def create_dataset(teacher,train_loader):  
        counter=0    
        cos_sim_counter=0   
        true_predicted_by_model = 0
        false_predicted_by_model = 0
        loop_counter = 0
        improved_dataset = {}
        teacher.eval()
        for batch_idx, (inputs, targets, idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing batches"):
            inputs, targets,idx  = inputs.to(device), targets.to(device), idx.to(device)
            outputs = teacher(inputs)              									
            predicted = outputs.argmax(1)
            predicted_probability = F.softmax(outputs, dim=1)

            # Filter indices where predicted probability exceeds the threshold
            above_threshold_mask = predicted_probability.max(dim=1)[0] >= args.threshold
         
            # Assign predicted labels for those above the threshold
            for i in torch.where(above_threshold_mask)[0]:  # Iterate over valid indices
                improved_dataset[idx[i].item()] = predicted[i].item()
                loop_counter = loop_counter + 1
                if predicted[i] == targets[i]:
                    true_predicted_by_model += 1                   

        #print(f'cos_sim pseudolabelling all/true: {cos_sim_counter}/{counter}: acc: {counter*100/cos_sim_counter:.2f}')
        print(f'predicted by model all/true: {loop_counter}/{true_predicted_by_model}, acc: {true_predicted_by_model*100/loop_counter:.2f}')
                               
        return improved_dataset


    
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


def test_idx(epoch,target,loader):
    test_loss = 0
    correct = 0
    total = 0
    misclassified_samples = []
    target.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets,idx) in enumerate(loader):
        
            inputs, targets, idx = inputs.to(device), targets.to(device), idx.to(device)
            outputs = target(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            misclassified = (predicted!= targets)
            #print(idx[misclassified])
            misclassified_samples.extend(idx[misclassified].tolist())
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 	
    return misclassified_samples

if args.budget == '4k':
    teach_cp = 'test/SubsTeach_trained_targetLabeledDataset_dataChosenwith_modelMLP_training_finetune_4k_test.pth' 
elif args.budget == '2k':
    teach_cp = 'test/SubsTeach_trained_targetLabeledDataset_dataChosenwith_modelMLP_training_finetune_2k_test.pth' 
elif args.budget == '1k':
    teach_cp = 'test/SubsTeach_trained_targetLabeledDataset_dataChosenwith_modelMLP_training_finetune_1k_test.pth'
elif args.budget == '250':
    teach_cp = 'test/SubsTeach_trained_targetLabeledDataset_dataChosenwith_modelMLP_training_finetune_250_test.pth'

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
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,10)
        )        
        
    def forward(self, x):
        x = self.encoder(x, out='h')
        #x = x.view(x.size(0), -1)  # Flatten the output of the encoder
        x = self.linear_classifier(x)   
        return x

encoder = EncodeProject()
encoder.fc = nn.Identity()
teacher = EncoderWithLinear(encoder)
checkpoint = torch.load(ckpt_path(teach_cp),map_location=device)
teacher.load_state_dict(checkpoint)
teacher = teacher.to(device)
for p in teacher.parameters():
    p.requires_grad = False

print("Test feature extractor")
test(1,teacher,test_loader)

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


final_dataset_dictionary = create_dataset(teacher,unlabeled_loader)

final_dataset_to_store = FinalDataset(trainset,final_dataset_dictionary, transform=None)

final_dataset = FinalDataset(trainset,final_dataset_dictionary, transform=transform_train)

# Concatenate the datasets
merged_dataset = torch.utils.data.ConcatDataset([final_dataset, transformed_target_labeled_dataset_for_final])
merged_dataset_to_store = torch.utils.data.ConcatDataset([final_dataset_to_store, target_labeled_dataset_for_final])

# Create a DataLoader for the merged dataset
merged_dataloader = DataLoader(merged_dataset, batch_size=128, shuffle=True, num_workers=2)

with open(out_path(f'transfer_dataset/{args.budget}_{args.threshold}_Whole_data_labeled_by_substitute_model_encoder+MLP.pkl'), 'wb') as f:
    pickle.dump(merged_dataset_to_store, f)

print("Final dataset with only chosen images has created and saved")

#print("length of finalized dataset",len(boost_dataset))
total = len(merged_dataset)
corr = 0
failed_image_num = 0
for _, label ,original_label in merged_dataloader:
    corr += (label == original_label).sum().item() 

failed_image_num  = total - corr
acc = corr*100/total
print(f"{corr} / {failed_image_num} Acc: %{corr*100/total:.2f}: true / false finazalized dataset distribution")


ckpt = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
new_enc = EncodeProject()
new_enc.load_state_dict(ckpt['state_dict'])
new_enc.fc = nn.Identity()
for p in new_enc.parameters():
    p.requires_grad = True
new_model= EncoderWithLinear(new_enc)
#checkpoint = torch.load(ckpt_path(teach_cp),map_location=device)
#teacher.load_state_dict(checkpoint)
new_model = new_model.to(device)

# Print trainable layers and their names
print("Trainable layers and their names:")
for name, param in new_model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

num_params = sum(p.numel() for p in teacher.parameters())
print(f"Number of parameters in teacher model: {num_params}")

opt_teacher = torch.optim.SGD(new_model.parameters(), lr=0.01,momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_teacher, T_max=20)
train(10,new_model,opt_teacher,merged_dataloader)

print("New model test acc is: ")
test(1,new_model,test_loader)
