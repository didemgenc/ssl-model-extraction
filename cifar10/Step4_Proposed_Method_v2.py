from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""In this code unlabaled data are sent to encoer+MLP model for pseudolabelling. If the model confidence score is under threshold than below_threshold samples are pseduolabed according 
to the most similar samples in the labeled domain. The difference of v2 from the v1 is that cos_sim is calculated on encoder+MLp model pseudolabeled samples and 4k target labeled samples both. 
however, in v1 only the 4k target """
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
from torch.utils.data import Dataset,DataLoader,ConcatDataset,Subset

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
"""
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
"""
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#parser.add_argument('--training_method', choices=['finetune', 'layer4','frozen'],required=True)
#parser.add_argument('--model', choices=['linear', 'MLP'],required=True)
#parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--substitute_threshold', choices=['0.8', '0.6','0.7'], required=True)
parser.add_argument('--budget', choices=['1k', '2k', '4k', '250'],required=True)
parser.add_argument('--cossim', choices=['target', 'substitute'],required=True)
parser.add_argument('--threshold',choices=['0.98','0.9', '0.95', '0.8', '0.7','0.6','0'],required=True)
args = parser.parse_args()

seed_value = 879
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(879)

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
    

trainset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True,transform=None)
testset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True,transform=None)

#transformed_target_labeled_dataset = TransformedDataset(target_labeled_dataset, transform=transform_train)
#target_labeled_loader = DataLoader(transformed_target_labeled_dataset, batch_size=128, shuffle=False, num_workers=2)

transformed_target_labeled_dataset_for_final = TransformedDataset(target_labeled_dataset_for_final, transform=transform_test)
transformed_target_labeled_loader = DataLoader(transformed_target_labeled_dataset_for_final, batch_size=64, shuffle=False, num_workers=2)

transformed_unlabeled_dataset = TransformedDataset(unlabeled_dataset, transform=transform_test)
unlabeled_loader = DataLoader(transformed_unlabeled_dataset, batch_size=64, shuffle=False, num_workers=2)

testset = TransformedDataset(test_dataset, transform=transform_test)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def create_dataset(teacher,train_loader):  
        true_predicted_by_model = 0
        loop_counter = 0
        improved_dataset_psudolabeled_by_model = {}
        unpseudolabeled_sample_indice = []
        teacher.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing batches"):
                inputs, targets,idx  = inputs.to(device), targets.to(device), idx.to(device)
                outputs = teacher(inputs)              									
                predicted = outputs.argmax(1)
                predicted_probability = F.softmax(outputs, dim=1)
        
                # Filter indices where predicted probability exceeds the threshold
                above_threshold_mask = predicted_probability.max(dim=1)[0] >= float(args.threshold)
                below_threshold_mask = ~above_threshold_mask  # Complement mask for below threshold

                # Assign predicted labels for those above the threshold
                for i in torch.where(above_threshold_mask)[0]:  # Iterate over valid indices
                    improved_dataset_psudolabeled_by_model[idx[i].item()] = predicted[i].item()
                    loop_counter += 1
                    if predicted[i] == targets[i]:
                        true_predicted_by_model += 1
                for i in torch.where(below_threshold_mask)[0]:  # Iterate over valid indices
                    unpseudolabeled_sample_indice.append(idx[i].item())
            
        #print(loop_counter)
        #print(true_predicted_by_model)
        #print(f'cos_sim pseudolabelling all/true: {cos_sim_counter}/{counter}: acc: {counter*100/cos_sim_counter:.2f}')
                                   
        return improved_dataset_psudolabeled_by_model, unpseudolabeled_sample_indice, loop_counter,true_predicted_by_model

from torch.nn.functional import cosine_similarity
from collections import Counter
"""
def find_most_similar(unlabeled_features, labeled_features):        
        labels=[]
        most_common_labels = []
        labeled_features = torch.tensor(labeled_features).to(device)
        unlabeled_features = torch.tensor(unlabeled_features).to(device)
        #print("labeled_features", labeled_features)
        labeled_len = labeled_features.size(1)-2   
        unlabeled_features_len = unlabeled_features.size(1)-2     
        unlabeled_features_normalized = unlabeled_features[:,:unlabeled_features_len] / unlabeled_features[:,:unlabeled_features_len].norm(dim=1, keepdim=True)
        labeled_features_nomalized = labeled_features[:,:labeled_len] / labeled_features[:,:labeled_len].norm(dim=1, keepdim=True)

        # Compute cosine similarity
        cos_sim = torch.mm(unlabeled_features_normalized, labeled_features_nomalized.T)
        print(cos_sim.size())
        top_value, top_indice = torch.topk(cos_sim, 1, dim=1)
        print("top_indice size",top_indice.size())
        print(top_value)
        for i, id in enumerate(top_indice):  
            most_common_labels.append([int(unlabeled_features[i,-2]), int(labeled_features[id, -1])])
            
            #for i in id:
            #    labels.append(int(labeled_features[i, labeled_len+1]))
            #element_label = Counter(labels)
            #most_common_label, count = element_label.most_common(1)[0]
            #most_common_labels.append(most_common_label)
            #labels.clear()
        #print("top_value",top_value)
        #print(int(labeled_features[top_indice, labeled_len]))
        #top_indice_id = int(labeled_features[top_indice.item(), labeled_len])
        #top_indice_label = int(labeled_features[top_indice, labeled_len+1])
        return most_common_labels
 
"""
def choose_images(tensor1,tensor2,top_indices):
    feature1_len = tensor1.size(1)-2
    #print("l_feature2",feature2_len)
    #print(top_indices.size(0))
    image_list = []
    for i, data in enumerate(top_indices):
        #print(data)
        if data.item() == 0:
            continue
        else:
            #for j in data:    
            image_list.append([int(tensor1[i][feature1_len]),int(tensor2[data.item()][-1])])         
    return image_list

def calculate_cosine_sim_loop(feature_1,feature_2): # unlabeled, labeled
    N_samples = 1
    batch_size=64
    similarity_matrix= torch.zeros(len(feature_1), len(feature_2))
    #print(similarity_matrix.size())
    tensor_f1 = torch.Tensor(feature_1)
    tensor_f2 = torch.Tensor(feature_2)
    feature1_len= tensor_f1.size(1)-2
    #print("l_feature1",feature1_len)
    feature2_len= tensor_f2.size(1)-2
    top_values_list = []
    top_indices_list = []
    # Step 1: Normalize each row of A and B (L2 normalization)
    A_normalized = tensor_f1[:,:feature1_len] / tensor_f1[:,:feature1_len].norm(dim=1, keepdim=True)
    B_normalized = tensor_f2[:, :feature2_len] / tensor_f2[:, :feature2_len].norm(dim=1, keepdim=True)
    print("calculting similarity matrix")
    #similarity_matrix = torch.mm(A_normalized, B_normalized.T) 
    num_samples_f1 = A_normalized.size(0)
    num_samples_f2 = B_normalized.size(0)
    print("Calculating similarity matrix in batches...")
    for i in range(0, num_samples_f1, batch_size):
        A_batch = A_normalized[i:i+batch_size]
        similarity_matrix_batch = torch.zeros(A_batch.size(0), num_samples_f2)

        for j in range(0, num_samples_f2, batch_size):
            B_batch = B_normalized[j:j+batch_size]
            similarity_matrix_batch[:, j:j+batch_size] = torch.mm(A_batch, B_batch.T)

        top_values_batch, top_indices_batch = torch.topk(similarity_matrix_batch, N_samples, dim=1)
        top_values_list.append(top_values_batch)
        top_indices_list.append(top_indices_batch)

    top_values = torch.cat(top_values_list, dim=0)
    top_indices = torch.cat(top_indices_list, dim=0)

    #print("sim_matrix",similarity_matrix_batch)
    #print("tensor1_list", tensor1_list)
    #print("similarity matrix lenght",len(similarity_matrix))
    #top_values, top_indices = torch.topk(similarity_matrix, N_samples, dim=1)
    #print(top_indices)
    # Print the top values and their indices
    threshold = float(args.substitute_threshold)
    mask = top_values <= threshold
    top_indices[mask] = 0
    #filtered_top_values = top_values[mask]
    #filtered_top_indices = top_indices[mask]
    print("Top values:", top_values)
    print("Top indices:", top_indices)
    print("Top indices size:", top_indices.size())
    #filtered_top_indices = filtered_top_indices.view(-1, N_samples)

    image_list = choose_images(tensor_f1,tensor_f2,top_indices)
    #print(len(image_list))
    print("images are chosen")
    return  image_list   

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
test_idx(1,teacher,test_loader)

def encode_loader(encoder,loader):
    encoded_data = []  
    for data, labels, idx in loader:
            data = data.to(device)
            encoded = encoder(data, out='h')
            for i,embd in enumerate(encoded):           
                encoded_data.append([embd.flatten().tolist(), idx[i].item(),labels[i].item()])
    encoded_data_flattened = [[item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])] for lst in encoded_data]    

    return encoded_data_flattened

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


samples_labeled_by_model, unpseudolabeled_sample_indices, loop_counter, true_predicted_by_model = create_dataset(teacher,unlabeled_loader)

print(f'predicted by substitute model all/true: {loop_counter}/{true_predicted_by_model}, acc: {true_predicted_by_model*100/loop_counter:.2f}')

samples_labeled_by_model_for_storing = PseudolabelByModel(trainset,samples_labeled_by_model, transform=None)

transformed_samples_labeled_by_model = PseudolabelByModel(trainset,samples_labeled_by_model, transform=transform_test)

transformed_samples_labeled_by_model_dataloader = DataLoader(transformed_samples_labeled_by_model,batch_size=64, shuffle=False, num_workers=2)
print("transformed samples labeled by model",len(transformed_samples_labeled_by_model))
print("transformed target labeled",len(transformed_target_labeled_dataset_for_final))
# Concatenate the datasets
"""

if args.threshold == '0':
    size=len(samples_labeled_by_model)
    if args.substitute_threshold == '0.6':
        with open(out_path(f'transfer_datasets/0.6substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(samples_labeled_by_model_for_storing, f)
    elif args.substitute_threshold == '0.7':
        with open(out_path(f'transfer_datasets/0.7substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(samples_labeled_by_model_for_storing, f)
    elif args.substitute_threshold == '0.8':
        with open(out_path(f'transfer_datasets/0.8substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
            pickle.dump(samples_labeled_by_model_for_storing, f)
"""
merged_dataset = torch.utils.data.ConcatDataset([transformed_samples_labeled_by_model, transformed_target_labeled_dataset_for_final])
print("total dataset length", len(merged_dataset))
merged_dataset_untransformed_for_storing = torch.utils.data.ConcatDataset([samples_labeled_by_model_for_storing, target_labeled_dataset_for_final])


with open(out_path(f'ablation7.pkl'), 'wb') as f:
            pickle.dump(merged_dataset_untransformed_for_storing, f)

# Create a DataLoader that will be used in cos_sim calculation as the labeled dataset!!
merged_dataloader = DataLoader(merged_dataset, batch_size=128, shuffle=False, num_workers=2)

#with open(out_path('transfer_dataset/samples_labeled_by_substitute_model.pkl'), 'wb') as f:
#    pickle.dump(samples_labeled_by_model_for_storing, f)

#with open(out_path('transfer_dataset/cos_sim_will_conducted_on_this_dataset.pkl'), 'wb') as f:
#    pickle.dump(merged_dataset_untransformed_for_storing, f)

print("Samples pseudolabeled by encoder+MLP model has created and saved")

#print("length of finalized dataset",len(boost_dataset))
total = len(merged_dataset)
corr = 0
failed_image_num = 0
for _, label ,original_label in merged_dataloader:
    corr += (label == original_label).sum().item() 

failed_image_num  = total - corr
acc = corr*100/total
print(f"{corr} / {failed_image_num} / {total} Acc: %{corr*100/total:.2f}: true / false distribution susbtitute labeled dataset!!")

class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]
        if self.transform:
            image = self.transform(image)
        return image, label, original_idx

#print(len(unpseudolabeled_sample_indices))
#print(unpseudolabeled_sample_indices)
#start to psedolabel the samples with cos_similarity
cos_sim_pseudolabel = CustomSubset(trainset, unpseudolabeled_sample_indices, transform = transform_test)
cos_sim_loader = DataLoader(cos_sim_pseudolabel, batch_size=32,shuffle=False)

unlabeled_features = encode_loader(teacher.encoder,cos_sim_loader) #unlabeled images which will be labeled by cos_sim
if args.cossim == 'substitute':
    labeled_features = encode_loader(teacher.encoder,merged_dataloader)    
elif args.cossim == 'target':
    labeled_features = encode_loader(teacher.encoder,transformed_target_labeled_loader)
 #merged_dataloader
print("embeddings are calculated, most similar labels will calculate")
similar_image_list = calculate_cosine_sim_loop(unlabeled_features,labeled_features)
#print(similar_image_list)

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
    
cos_sim_labeled_dataset = CosSimLabeledDataset(trainset, similar_image_list, transform = transform_test)
untransformed_cos_sim_labeled_dataset_for_storing = CosSimLabeledDataset(trainset, similar_image_list, transform = None)
cos_sim_labeled_dataloader = DataLoader(cos_sim_labeled_dataset, batch_size=64,shuffle=False)

#print("length of finalized dataset",len(boost_dataset))
total_cossim = len(cos_sim_labeled_dataset)
corr_cossim = 0
failed_image_num_cossim  = 0
for _, label ,original_label in cos_sim_labeled_dataloader:
    corr_cossim += (label == original_label).sum().item() 

failed_image_num_cossim = total_cossim - corr_cossim
acc_cossim = corr_cossim*100/total_cossim
print(f"{corr_cossim} / {failed_image_num_cossim} / {total_cossim} Acc: %{corr_cossim*100/total_cossim:.2f}: true / false distribution cos_sim labeled dataset!!")

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

final_dataset = CombinedDataset(merged_dataset_untransformed_for_storing, untransformed_cos_sim_labeled_dataset_for_storing)

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

transformed_final_dataset = FinalTransformedDataset(final_dataset, transform=transform_test)

# Create a DataLoader for the merged dataset
final_dataloader = DataLoader(transformed_final_dataset, batch_size=64, shuffle=False, num_workers=2)

size= len(cos_sim_labeled_dataset)
if args.substitute_threshold=='0.6':
   with open(out_path(f'transfer_dataset/0.6substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
        pickle.dump(final_dataset, f)
elif args.substitute_threshold=='0.7':
   with open(out_path(f'transfer_dataset/0.7substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
        pickle.dump(final_dataset, f)
elif args.substitute_threshold=='0.8':
   with open(out_path(f'transfer_dataset/0.8substitute_threshold/{args.budget}_{args.threshold}_confidence+cossim_labeled_dataset.pkl'), 'wb') as f:
        pickle.dump(final_dataset, f)

#print("length of finalized dataset",len(boost_dataset))
total_final= len(transformed_final_dataset)
corr_final = 0
failed_image_num_final= 0
for _, label ,original_label in final_dataloader:
    corr_final+= (label == original_label).sum().item() 

failed_image_num_final  = total_final - corr_final
acc_final = corr_final*100/total_final
print(f"{corr_final} / {failed_image_num_final} / {total_final} Acc: %{corr_final*100/total_final:.2f}: true / false finazalized dataset distribution")
end_time = time.time()
# End time
# Calculate the execution time
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
print(f"Execution time: {execution_time_seconds:.2f} seconds ({execution_time_minutes:.2f} minutes)")

if args.substitute_threshold == '0.6':
    # Write budget and threshold to a file
    with open('transfer_dataset_accuracies_0.6_subs_threshold.txt', 'a') as file:
        file.write(f"Budget: {args.budget}\t")
        file.write(f"Subs_threshold: {args.substitute_threshold}\t")
        file.write(f"Threshold: {args.threshold}\n")
        file.write(f"predicted by substitute: {true_predicted_by_model}/{loop_counter-true_predicted_by_model}/{loop_counter} Acc: %{true_predicted_by_model*100/loop_counter:.2f}\n")
        file.write(f"{corr} / {failed_image_num} / {total} Acc: %{corr*100/total:.2f}: true / false distribution susbtitute+ target labeled dataset!!\n")
        file.write(f"{corr_cossim} / {failed_image_num_cossim} / {total_cossim} Acc: %{corr_cossim*100/total_cossim:.2f}: true / false distribution cos_sim labeled dataset!!\n")
        file.write(f"{corr_final} / {failed_image_num_final} / {total_final} Acc: %{corr_final*100/total_final:.2f}: true / false finazalized dataset distribution\n")

elif args.substitute_threshold == '0.7':
    # Write budget and threshold to a file
    with open('transfer_dataset_accuracies_0.7_subs_threshold.txt', 'a') as file:
        file.write(f"Budget: {args.budget}\t")
        file.write(f"Subs_threshold: {args.substitute_threshold}\t")
        file.write(f"Threshold: {args.threshold}\n")
        file.write(f"predicted by substitute: {true_predicted_by_model}/{loop_counter-true_predicted_by_model}/{loop_counter} Acc: %{true_predicted_by_model*100/loop_counter:.2f}\n")
        file.write(f"{corr} / {failed_image_num} / {total} Acc: %{corr*100/total:.2f}: true / false distribution susbtitute+ target labeled dataset!!\n")
        file.write(f"{corr_cossim} / {failed_image_num_cossim} / {total_cossim} Acc: %{corr_cossim*100/total_cossim:.2f}: true / false distribution cos_sim labeled dataset!!\n")
        file.write(f"{corr_final} / {failed_image_num_final} / {total_final} Acc: %{corr_final*100/total_final:.2f}: true / false finazalized dataset distribution\n")

elif args.substitute_threshold == '0.8':
    # Write budget and threshold to a file
    with open('transfer_dataset_accuracies_0.8_subs_threshold.txt', 'a') as file:
        file.write(f"Budget: {args.budget}\t")
        file.write(f"Subs_threshold: {args.substitute_threshold}\t")
        file.write(f"Threshold: {args.threshold}\n")
        file.write(f"predicted by substitute: {true_predicted_by_model}/{loop_counter-true_predicted_by_model}/{loop_counter} Acc: %{true_predicted_by_model*100/loop_counter:.2f}\n")
        file.write(f"{corr} / {failed_image_num} / {total} Acc: %{corr*100/total:.2f}: true / false distribution susbtitute+ target labeled dataset!!\n")
        file.write(f"{corr_cossim} / {failed_image_num_cossim} / {total_cossim} Acc: %{corr_cossim*100/total_cossim:.2f}: true / false distribution cos_sim labeled dataset!!\n")
        file.write(f"{corr_final} / {failed_image_num_final} / {total_final} Acc: %{corr_final*100/total_final:.2f}: true / false finazalized dataset distribution\n")


"""
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
"""