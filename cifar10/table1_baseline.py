"""Reproduce Table 1: semi-supervised baseline extracted model on CIFAR-10.

Parameterised rewrite of Encoder_Finetune_Eval_with_target_labeled_dataset.py.
Training recipe is kept identical (SimCLR ResNet-50 encoder, SGD lr 0.1 momentum 0.9,
batch 128, 20 epochs, random-crop + flip augmentation). Differences:
  * --budget, --seed are arguments instead of hard-coded values
  * both the best-epoch test accuracy (what the original script tracked) and the
    final-epoch test accuracy (no test-set model selection) are recorded
  * one CSV row per run is appended to outputs/table1_results.csv
  * checkpoint name carries budget, model, method and seed
"""
from paths import data_path, out_path, ckpt_path

import argparse
import csv
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from models.encoder import EncodeProject

parser = argparse.ArgumentParser(description='Table 1 baseline extracted model')
parser.add_argument('--budget', choices=['4k', '2k', '1k', '250'], required=True)
parser.add_argument('--training_method', choices=['finetune', 'layer4', 'frozen'], required=True)
parser.add_argument('--model', choices=['linear', 'MLP'], required=True)
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--seed', default=950, type=int)
parser.add_argument('--results', default='table1_results.csv')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_time = time.time()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ---- dataset classes: names must match the ones the pickles were written with ----
class ModifiedLabelsDataset(Dataset):
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


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

with open(data_path(f'{args.budget}_target_labeled_dataset.pkl'), 'rb') as f:
    transfer_dataset = pickle.load(f)
with open(data_path(f'random_{args.budget}_test_dataset.pkl'), 'rb') as f:
    test_dataset = pickle.load(f)

# sanity: query set and test set must be disjoint
overlap = set(transfer_dataset.filtered_indices) & set(test_dataset.indexes)
print(f'budget {args.budget}: {len(transfer_dataset)} query samples, {len(test_dataset)} test samples, overlap {len(overlap)}')

transfer_loader = DataLoader(TransformedDataset(transfer_dataset, transform_train), batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(TransformedDataset(test_dataset, transform_test), batch_size=128, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()


class EncoderWithLinear(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        if head == 'linear':
            self.linear_classifier = nn.Linear(2048, 10)
        else:
            self.linear_classifier = nn.Sequential(
                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(512, 10),
            )

    def forward(self, x):
        return self.linear_classifier(self.encoder(x, out='h'))


ckpt = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
encoder = EncodeProject()
encoder.load_state_dict(ckpt['state_dict'])
encoder.fc = nn.Identity()
if args.training_method == 'finetune':
    for p in encoder.parameters():
        p.requires_grad = True
elif args.training_method == 'layer4':
    for name, param in encoder.named_parameters():
        param.requires_grad = 'layer4' in name
elif args.training_method == 'frozen':
    for p in encoder.parameters():
        p.requires_grad = False
teacher = EncoderWithLinear(encoder, args.model).to(device)

num_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
print(f'trainable parameters: {num_trainable}')

opt = torch.optim.SGD(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, momentum=0.9)
teach_cp = f'table1/{args.budget}_{args.model}_{args.training_method}_seed{args.seed}.pth'


def evaluate(model):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_sum += criterion(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total, loss_sum / len(test_loader)


best_acc, best_epoch = 0.0, -1
for epoch in range(args.epoch):
    teacher.train()
    train_loss = correct_train = total_train = 0
    for inputs, targets, _ in transfer_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        outputs = teacher(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        correct_train += (outputs.argmax(1) == targets).sum().item()
        total_train += targets.size(0)

    test_acc, test_loss = evaluate(teacher)
    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(teacher.state_dict(), ckpt_path(teach_cp))
    print('(%d/%d) Train Loss: %.3f  Acc: %.3f%% (%d/%d) | Test Loss: %.3f  Acc: %.3f%%'
          % (epoch, args.epoch, train_loss / len(transfer_loader), 100.0 * correct_train / total_train,
             correct_train, total_train, test_loss, test_acc))

final_acc, _ = evaluate(teacher)
torch.save(teacher.state_dict(), ckpt_path(teach_cp.replace('.pth', '_final.pth')))
elapsed = (time.time() - start_time) / 60
print(f'RESULT budget={args.budget} model={args.model} method={args.training_method} seed={args.seed} '
      f'best_acc={best_acc:.2f} (epoch {best_epoch}) final_acc={final_acc:.2f} time={elapsed:.1f}min')

results_file = out_path(args.results)
write_header = not os.path.exists(results_file)
with open(results_file, 'a', newline='') as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(['budget', 'model', 'training_method', 'seed', 'epochs', 'lr',
                    'best_test_acc', 'best_epoch', 'final_test_acc', 'n_query', 'n_test', 'minutes'])
    w.writerow([args.budget, args.model, args.training_method, args.seed, args.epoch, args.lr,
                f'{best_acc:.2f}', best_epoch, f'{final_acc:.2f}', len(transfer_dataset), len(test_dataset), f'{elapsed:.1f}'])
