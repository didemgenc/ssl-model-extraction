from paths import DATASET_ROOT, data_path, out_path, ckpt_path
"""
This code upload the pretrained encoder network and the target labeled dataset.
By using cosine similarity metric, the most similar n images are chosen among the unlabeled pool.
Chosen images are labeled with the label's of the most similar  target image.
Finally, the transfer datase is obtained by merging the target labeled dataset and the chosen images. 
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
from models.wideresnet import Wide_ResNet

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--n_samples', default=2, type=int)
parser.add_argument('--label_whole', default=1, type=int)
parser.add_argument('--threshold', default=0.9, type=float)
parser.add_argument('--method', choices=['AL', 'random'],required=True)
#parser.add_argument('--lr',default=0.1, type=float)
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
    
seed_value = 950
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.random.manual_seed(950)

criterion = nn.CrossEntropyLoss()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.method == 'AL':
    with open(data_path('transfer_dataset/AL_4k_target_labeled_dataset.pkl'), 'rb') as f:
        target_labeled_dataset = pickle.load(f)
    with open(data_path('transfer_dataset/AL_unlabeled_dataset.pkl'), 'rb') as f:
        unlabeled_dataset = pickle.load(f)
    with open(data_path('transfer_dataset/AL_val_dataset.pkl'), 'rb') as f:
        val_dataset = pickle.load(f)    
    
elif args.method == 'random':
    with open(data_path('transfer_dataset/random_4k_target_labeled_dataset.pkl'), 'rb') as f:
        target_labeled_dataset = pickle.load(f)
    with open(data_path('transfer_dataset/random_unlabeled_dataset.pkl'), 'rb') as f:
        unlabeled_dataset = pickle.load(f)
    with open(data_path('transfer_dataset/random_val_dataset.pkl'), 'rb') as f:
        val_dataset = pickle.load(f)

trainset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True,transform=None)

transformed_target_labeled_dataset = TransformedDataset(target_labeled_dataset, transform=transform_test)
target_labeled_loader = DataLoader(transformed_target_labeled_dataset, batch_size=64, shuffle=False, num_workers=2)

transformed_unlabeled_dataset = TransformedDataset(unlabeled_dataset, transform=transform_test)
unlabeled_loader = DataLoader(transformed_unlabeled_dataset, batch_size=64, shuffle=False, num_workers=2)

transformed_val_dataset = TransformedDataset(val_dataset, transform=transform_test)
val_loader = DataLoader(transformed_val_dataset, batch_size=64, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True,transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

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


def encode_loader(encoder,loader):
    encoded_data = []  
    for data, labels, idx in loader:
            data = data.to(device)
            encoded = encoder(data, out='h')
            for i,embd in enumerate(encoded):           
                encoded_data.append([embd.flatten().tolist(), idx[i].item(),labels[i].item()])
    encoded_data_flattened = [[item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])] for lst in encoded_data]    
    
    #if you train a linear model on top of encoder, use this code
    #dataset = TensorDataset(encoded_data, encoded_labels)
    #return DataLoader(dataset, batch_size=loader.batch_size)
    return encoded_data_flattened

#teach_cp = "/home/didemgenc/Desktop/2024_SSL_MEA/Scenario4_MEA_SSL_KD/teacher_cp/subsTeach_4kbudget_encoder_finetuning_MLP_Eval.pth"
"""
class EncoderWithLinear(nn.Module):
    def __init__(self, encoder):
        super(EncoderWithLinear, self).__init__()
        self.encoder = encoder 
        #For linear eval use this
        self.linear_classifier = nn.Linear(2048, 10)
        '''
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
        
        '''
    def forward(self, x):
        with torch.no_grad(): 
            x = self.encoder(x, out='h')
        #x = x.view(x.size(0), -1)  # Flatten the output of the encoder
        x = self.linear_classifier(x)   
        return x
"""
ckpt = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
feature_extractor = EncodeProject()
feature_extractor.load_state_dict(ckpt['state_dict'])
feature_extractor.fc = nn.Identity()
for p in feature_extractor.parameters():
    p.requires_grad = False
#teacher = EncoderWithLinear(encoder)
#checkpoint = torch.load(ckpt_path(teach_cp),map_location=device)
#teacher.load_state_dict(checkpoint)
feature_extractor = feature_extractor.to(device)

print("Extracting embeddings...")
train_embeddings = encode_loader(feature_extractor,target_labeled_loader)
unlabeled_pool_embeddings = encode_loader(feature_extractor,unlabeled_loader)

print("Embeddings are extracted")

# import required libraries
import numpy as np
from numpy.linalg import norm

def remove_redundant_elements(list_of_lists):
    unique_elements = set()
    new_list_of_lists = []

    for sublist in list_of_lists:
        new_sublist = []
        for elem in sublist:
            if elem not in unique_elements:
                unique_elements.add(elem)
                new_sublist.append(elem)
        new_list_of_lists.append(new_sublist)

    return new_list_of_lists
from collections import Counter
def choose_images(tensor1,tensor_2,top_indices):
    feature2_len= tensor_2.size(1)-2
   
    #print("l_feature2",feature2_len)
    #print(top_indices.size(0))
    image_list = []
    labels=[]
    if args.label_whole == 0:
        for i, data in enumerate(top_indices):
            image_list.append([int(tensor1[i][0]),int(tensor1[i][1])])
            for j in data:
                    if j.item() != 0: 
                        #print(tensor_2[j][feature2_len])          
                        image_list.append([int(tensor_2[j][feature2_len]),int(tensor1[i][1])])
    elif args.label_whole == 1:
        top_indices= top_indices.T
        print(top_indices)
        for i,data in enumerate(top_indices):
            #in order to assign pseudolabel which is the most repetitive one in 11 samples, then uncomment the following commented lines
            #print("data",data)
            for j, id in enumerate(data):
            #    labels.append(int(tensor1[id][1]))
            #element_label = Counter(labels)
            #most_common_label, count = element_label.most_common(1)[0]
            #labels.clear()
            #print(classes[most_common_label])
            #image_list.append([int(tensor_2[i][feature2_len]),most_common_label])
                image_list.append([int(tensor_2[i][feature2_len]),int(tensor1[id][1])])
        for i in range(len(tensor1)):
            image_list.append([int(tensor1[i][0]),int(tensor1[i][1])])
       
    return image_list

def calculate_cosine_sim_loop(feature_1,feature_2):
    N_samples = args.n_samples
    similarity_matrix= torch.zeros(len(feature_1), len(feature_2))
    print(similarity_matrix.size())
    tensor1_list = []
    tensor_f1 = torch.Tensor(feature_1)
    tensor_f2 = torch.Tensor(feature_2)
    feature1_len= tensor_f1.size(1)-2
    #print("l_feature1",feature1_len)
    feature2_len= tensor_f2.size(1)-2
    #print("l_feature2",feature2_len)
    #for i in tqdm(range(tensor_f1.size(0)),desc = "Calculating similarity matrix"):
    #        tensor1_list.append([int(tensor_f1[i][-2]),int(tensor_f1[i][-1])])
    #        cosine = np.dot(tensor_f2[:,:feature2_len],tensor_f1[i][:feature1_len])/(norm(tensor_f2[:, :feature2_len], axis=1)*norm(tensor_f1[i][:feature1_len]))
    #        similarity_matrix[i] = torch.from_numpy(cosine).unsqueeze(0)

    #faster calculation
    for i in tqdm(range(tensor_f1.size(0)), desc="Calculating similarity matrix"):
        tensor1_list.append([int(tensor_f1[i][-2]), int(tensor_f1[i][-1])])
    # Step 1: Normalize each row of A and B (L2 normalization)
    A_normalized = tensor_f1[:,:feature1_len] / tensor_f1[:,:feature1_len].norm(dim=1, keepdim=True)
    B_normalized = tensor_f2[:, :feature2_len] / tensor_f2[:, :feature2_len].norm(dim=1, keepdim=True)

    similarity_matrix = torch.mm(A_normalized, B_normalized.T) 
    print("sim_matrix",similarity_matrix)
    #print("tensor1_list", tensor1_list)
    #print("similarity matrix lenght",len(similarity_matrix))
    if args.label_whole == 0:
        top_values, top_indices = torch.topk(similarity_matrix, N_samples, dim=1)
        # Print the top values and their indices
        #mask = top_values >= 0.9
        #filtered_indices = top_indices[mask]
        print("Top values:", top_values)
        print("Top indices size:", top_indices.size())
        threshould = args.threshold
        # Create a mask where each element is True if the corresponding element in top_values is greater than or equal to 0.9, and False otherwise
        mask = top_values >= threshould

        # Use the mask to replace the elements in top_indices where the mask is False with 0
        top_indices = top_indices * mask.long()
        print("top_indices eliminated under threshold",top_indices)

        remove_redundant = remove_redundant_elements(top_indices)
        #print("redundants are eliminated",len(remove_redundant))

        image_list = choose_images(tensor1_list,tensor_f2,remove_redundant)

    elif args.label_whole == 1:
        top_values, top_indices = torch.topk(similarity_matrix,N_samples,dim=0)
        print("Top values:", top_values)
        print("Top indices size:", top_indices.size())
        print(type(top_indices))
        image_list = choose_images(tensor1_list,tensor_f2,top_indices)
        print(len(image_list))


    print("images are chosen")
    return  image_list 

#nodes, _ = get_graph_node_names(substitute_model)
#print(nodes)
        
#train_embeddings among the testset, unlabeled_pool_embeddings among the trainset of cifar10 dataset
similar_image_list = calculate_cosine_sim_loop(train_embeddings, unlabeled_pool_embeddings)
print("len of image list",len(similar_image_list))

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
     
#Get noisy labels!!
improved_dataset = ImprovedDataset(trainset, similar_image_list)
transformed_improved_dataset = TransformedDataset(improved_dataset, transform=transform_test)
boost_dataloader = torch.utils.data.DataLoader(transformed_improved_dataset, batch_size =128, shuffle=False,num_workers=2)

if args.label_whole == 0:
    size = len(improved_dataset) // 1000
    # Save the dataset as a .pkl file
    with open(out_path(f'{args.method}_threshold_{args.threshold}_Nsampling_{args.n_samples}_size_{size}k_cossim_chosen_dataset.pkl'), 'wb') as f:
        pickle.dump(improved_dataset, f)
elif args.label_whole == 1:
    size = len(improved_dataset) // 1000
    with open(out_path(f'Whole_data_labeled_cossim_{size}k.pkl'), 'wb') as f:
        pickle.dump(improved_dataset, f)

print("dataset stored!!")

#print("length of finalized dataset",len(boost_dataset))
total = len(improved_dataset)
corr = 0
failed_image_num = 0
for _, label ,original_label in boost_dataloader:
    corr += (label == original_label).sum().item() 

failed_image_num  = total - corr
acc = corr*100/total
print(f"{corr} / {failed_image_num} Acc: %{corr*100/total:.2f}: true / false finazalized dataset distribution")

filename = 'results.txt'

# Write the arguments and result to the file
with open(filename, 'a') as file:
    file.write(f"threshold: {args.threshold}\t")
    file.write(f"n_samples: {args.n_samples}\t")
    file.write(f"total: {total}\t")
    file.write(f"Acc: {acc}\n")

"""
def imshow(img):
    #img = img.permute(1, 2, 0) 
    img = img.float() 
    img = img * 0.2435 + 0.4822  # Unnormalize
    img = img.clamp(0, 1)  # Clamp the values to be between 0 and 1
    #npimg = img.numpy()    
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Display the image
    plt.show()
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
            scheduler.step()

            loss_epoch += loss.item()

            valid_loss = 0.0
            correct_val=0
            total_val=0
            model.eval()      
            for data, labels in val_loader:
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
        
        #scheduler_ft.step()                 
                         
        print('(%d/%d) Linear Model Train Loss: %.3f\t Acc: %.3f%% (%d/%d) |  Valid Loss: %.3f\t Acc: %.3f%% (%d/%d)'
		                 % (epoch,num_epochs,loss_epoch/(step+1), 100.*correct_train/total_train, correct_train,total_train, valid_loss/(step+1), 100.*correct_val/total_val, correct_val, total_val))
'''
if args.retrain==False:
    if args.model == 'Resnet18':
        model = resnet18(weights=None) 
        for params in model.parameters():
                    params.requires_grad_ = True                
        num_ftrs_model= model.fc.in_features                
        model.fc = nn.Linear(num_ftrs_model,10)   
        model.to(device)
        cp = "R18_subsstd_data_cosine_sim_image_chosen.pth"

    elif args.model == 'Resnet50':  
        model = resnet50(weights=None) 
        for params in model.parameters():
                    params.requires_grad_ = True                
        num_ftrs_model= model.fc.in_features                
        model.fc = nn.Linear(num_ftrs_model,10)   
        model.to(device) 
        cp = "R50_subsstd_data_cosine_sim_image_chosen.pth"

    elif args.model == 'Wideresnet': 
        model = Wide_ResNet(28, 2, 0, 10)
        model.to(device)
        cp = "WideRN_subsstd_data_cosine_sim_image_chosen.pth"
elif args.retrain==True:
    encoder_retrain = EncodeProject()
    encoder_retrain.fc = nn.Identity()
    model = EncoderWithLinear(encoder_retrain)
    checkpoint2 = torch.load(ckpt_path(teach_cp),map_location=device)
    model.load_state_dict(checkpoint2)
    model = model.to(device)
    # Set requires_grad to True for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    print("Before retraining test acc:")
    test(1,model,testloader)
    cp = "Retrained_EncoderMLP_subsstd_data_cosine_sim_image_chosen.pth"

# Calculate the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in {args.model}: {num_params}")
'''
ckpt_encoder = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
rt_encoder = EncodeProject()
rt_encoder.load_state_dict(ckpt_encoder['state_dict'])
rt_encoder.fc = nn.Identity()
for p in rt_encoder.parameters():
    p.requires_grad = True
substitute_model = EncoderWithLinear(rt_encoder)
#checkpoint = torch.load(ckpt_path(teach_cp),map_location=device)
#teacher.load_state_dict(checkpoint)
substitute_model = substitute_model.to(device)

cp = "After_enhancing_targetlabeleddataset_encoder_finetuned_with_LinearModel.pth"

# Print trainable layers and their names
print("Trainable layers and their names:")
for name, param in substitute_model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

num_params = sum(p.numel() for p in substitute_model.parameters())
print(f"Number of parameters in teacher model: {num_params}")

print(f'opt: {args.opt}, lr: {args.lr}, epoch: {args.epoch}, n_samples: {args.n_samples}')
opt_model= torch.optim.SGD(substitute_model.parameters(), lr=args.lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=20)
criterion = torch.nn.CrossEntropyLoss()
train(args.epoch,boost_dataloader,valloader,substitute_model,criterion,opt_model,cp)
test(1,substitute_model,testloader)

"""
