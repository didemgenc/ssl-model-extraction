"""Proposed method in the Table 3 / MixMatch setting (queries from the CIFAR-10 TRAINING set, full 10k test set).

Same recipe as table2_proposed.py (Step5), but the training set is the two-stage pseudo-labelled dataset of the
"original" experiment folder (class-balanced random queries drawn from the training set, as in MixMatch):
  transfer_dataset/data_labeled_with_cossim_and_substitute/{sim}similarity_threshold/private_{budget}_{conf}_confidence+cossim_labeled_dataset.pkl
and evaluation is on the FULL CIFAR-10 test set (10k images, victim accuracy 94.40%), reporting accuracy and
fidelity (agreement with the victim) of the final-epoch model. Results -> outputs/table3_results.csv,
checkpoints -> table3/{budget}_conf{conf}_sim{sim}_seed{seed}[_final].pth.

Original (Table 2) docstring follows.

Parameterised rewrite of Step5_TrainAndEval_SusbtituteModel.py. Trains the final
substitute model (SimCLR ResNet-50 encoder + 4-layer MLP head, full fine-tuning) on the
two-stage pseudo-labelled dataset produced by Step4:
  transfer_dataset/{sim}substitute_threshold/{budget}_{conf}_confidence+cossim_labeled_dataset.pkl
Recipe kept identical to exp_step5_0.7subs_threshold.sh: SGD lr 0.01, momentum 0.9,
weight decay 1e-3, batch 128, 10 epochs, no scheduler, random-crop + flip.
Records best-epoch (original script) and final-epoch (no test-set selection) accuracy and
appends one row to outputs/table2_results.csv. Checkpoints:
  table2/{budget}_conf{conf}_sim{sim}_seed{seed}.pth and ..._final.pth
Paper settings: 4k conf 0.9, 2k conf 0.9, 1k conf 0.95, 250 conf 0.95, sim 0.7.
"""
from paths import data_path, out_path, ckpt_path, orig_path, DATASET_ROOT
import torchvision

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

parser = argparse.ArgumentParser(description='Table 2 proposed method (Step5 with seed)')
parser.add_argument('--budget', choices=['4k', '2k', '1k', '250'], required=True)
parser.add_argument('--conf', default=None, help='confidence threshold used in Step4 (0.9, 0.95, ...); default = paper setting for the budget')
parser.add_argument('--sim', default='0.8', choices=['0.6', '0.8'], help='similarity (substitute) threshold used in Step4 (only 0.6 and 0.8 exist in this setup)')
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--scheduler', default=0, type=int)
parser.add_argument('--seed', default=950, type=int)
parser.add_argument('--results', default='table3_results.csv')
args = parser.parse_args()
PAPER_CONF = {'4k': '0.9', '2k': '0.9', '1k': '0.95', '250': '0.95'}
conf = args.conf or PAPER_CONF[args.budget]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_time = time.time()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ---- dataset classes: names must match the ones the pickles were written with ----
class CustomDataset(Dataset):
    def __init__(self, trainset, indexes, transform=None):
        self.trainset, self.indexes, self.transform = trainset, indexes, transform

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
        self.original_dataset, self.transform = original_dataset, transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        image, label, original_idx = self.original_dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label, original_idx


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1, self.dataset2 = dataset1, dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx - len(self.dataset1)]


class CosSimLabeledDataset(Dataset):
    def __init__(self, dataset, id_list, transform=None):
        self.dataset, self.id_list, self.transform = dataset, id_list, transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id, label = self.id_list[idx]
        image, original_label = self.dataset[id]
        if self.transform:
            image = self.transform(image)
        return image, label, original_label


class PseudolabelByModel(Dataset):
    def __init__(self, original_dataset, target_labels_dict, transform=None):
        self.original_dataset, self.target_labels_dict, self.transform = original_dataset, target_labels_dict, transform
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        image, original_label = self.original_dataset[original_idx]
        if self.transform:
            image = self.transform(image)
        return image, self.target_labels_dict.get(original_idx, -1), original_label


class ModifiedLabelsDatasetforFinalDatasetConcat(Dataset):
    def __init__(self, original_dataset, target_labels_dict):
        self.original_dataset, self.target_labels_dict = original_dataset, target_labels_dict
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        image, original_label = self.original_dataset[original_idx]
        return image, self.target_labels_dict.get(original_idx, -1), original_label


ModifiedLabelsDataset = ModifiedLabelsDatasetforFinalDatasetConcat
ImprovedDataset = CosSimLabeledDataset
FinalDataset = PseudolabelByModel


class FinalTransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, original_label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, original_label


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

train_pkl = f'transfer_dataset/data_labeled_with_cossim_and_substitute/{args.sim}similarity_threshold/private_{args.budget}_{conf}_confidence+cossim_labeled_dataset.pkl'
with open(orig_path(train_pkl), 'rb') as f:
    train_dataset = pickle.load(f)
test_base = torchvision.datasets.CIFAR10(DATASET_ROOT, train=False, download=False)
test_dataset = CustomDataset(test_base, list(range(len(test_base))))  # full 10k test set

train_loader = DataLoader(FinalTransformedDataset(train_dataset, transform_train), batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(TransformedDataset(test_dataset, transform_test), batch_size=128, shuffle=False, num_workers=2)

# pseudo-label quality of the training set (same diagnostic as the original script)
corr = sum((lab == gt).sum().item() for _, lab, gt in DataLoader(FinalTransformedDataset(train_dataset, transform_test), batch_size=512, num_workers=2))
n_train = len(train_dataset)
print(f'budget {args.budget} conf {conf} sim {args.sim}: {n_train} training samples, pseudo-label accuracy {100.0 * corr / n_train:.2f}%, {len(test_dataset)} test samples')


class EncoderWithMLP(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.linear_classifier = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.linear_classifier(self.encoder(x, out='h'))


from models.wideresnet import Wide_ResNet  # noqa: F401,E402  (needed to unpickle the victim)
victim = torch.load(data_path('wide-resnet-40x10.t7'), map_location=device, weights_only=False)['net'].to(device).eval()
with torch.no_grad():
    victim_pred = torch.cat([victim(x.to(device)).argmax(1).cpu() for x, _, _ in test_loader])
test_true = torch.tensor([y for _, y, _ in TransformedDataset(test_dataset)])
print(f'victim accuracy on the full test set: {100.0 * (victim_pred == test_true).float().mean():.2f}%')
del victim
torch.cuda.empty_cache()

ckpt = torch.load(data_path('resnet50_cifar10_bs1024_epochs1000.pth.tar'), map_location=device)
encoder = EncodeProject()
encoder.load_state_dict(ckpt['state_dict'])
encoder.fc = nn.Identity()
model = EncoderWithMLP(encoder).to(device)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6) if args.scheduler else None
cp = f'table3/{args.budget}_conf{conf}_sim{args.sim}_seed{args.seed}.pth'


def evaluate(m, with_fidelity=False):
    m.eval()
    preds = []
    with torch.no_grad():
        for inputs, _, _ in test_loader:
            preds.append(m(inputs.to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    acc = 100.0 * (preds == test_true).float().mean().item()
    return (acc, 100.0 * (preds == victim_pred).float().mean().item()) if with_fidelity else acc


best_acc, best_epoch = 0.0, -1
for epoch in range(args.epoch):
    model.train()
    loss_sum = correct_train = total_train = 0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        correct_train += (out.argmax(1) == y).sum().item()
        total_train += y.size(0)
    if scheduler:
        scheduler.step()
    test_acc = evaluate(model)
    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(model.state_dict(), ckpt_path(cp))
    print('(%d/%d) Train Loss: %.3f  Acc: %.3f%% | Test Acc: %.3f%%'
          % (epoch, args.epoch, loss_sum / len(train_loader), 100.0 * correct_train / total_train, test_acc))

final_acc, final_fid = evaluate(model, with_fidelity=True)
torch.save(model.state_dict(), ckpt_path(cp.replace('.pth', '_final.pth')))
elapsed = (time.time() - start_time) / 60
print(f'RESULT budget={args.budget} conf={conf} sim={args.sim} seed={args.seed} '
      f'best_acc={best_acc:.2f} (epoch {best_epoch}) final_acc={final_acc:.2f} fidelity={final_fid:.2f} time={elapsed:.1f}min')

results_file = out_path(args.results)
write_header = not os.path.exists(results_file)
with open(results_file, 'a', newline='') as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(['budget', 'conf', 'sim', 'seed', 'epochs', 'lr', 'scheduler', 'best_test_acc', 'best_epoch',
                    'final_test_acc', 'final_fidelity', 'n_train', 'pseudo_label_acc', 'n_test', 'minutes'])
    w.writerow([args.budget, conf, args.sim, args.seed, args.epoch, args.lr, args.scheduler, f'{best_acc:.2f}', best_epoch,
                f'{final_acc:.2f}', f'{final_fid:.2f}', n_train, f'{100.0 * corr / n_train:.2f}', len(test_dataset), f'{elapsed:.1f}'])
