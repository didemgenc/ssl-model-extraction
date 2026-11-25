"""Evaluate extracted-model checkpoints against the victim: accuracy and fidelity.

Fidelity follows Jagielski et al. (USENIX Sec. 2020): fraction of test samples on which
the extracted model predicts the same label as the victim. Evaluation is on the same
6000-image held-out split used for accuracy (random_{budget}_test_dataset.pkl).

Usage:
  python eval_fidelity.py --glob 'table1/*_seed*.pth'            # all Table 1 checkpoints
  python eval_fidelity.py --glob 'table1/4k_linear_finetune_seed950_final.pth'
  python eval_fidelity.py --glob 'table2/*.pth' --results fidelity_table2.csv   # Table 2 proposed method
Appends one row per checkpoint to outputs/fidelity_results.csv.
"""
from paths import data_path, out_path, ckpt_path

import argparse
import csv
import glob
import os
import pickle
import re

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from models.encoder import EncodeProject
from models.wideresnet import Wide_ResNet  # noqa: F401  (needed to unpickle the victim checkpoint)

parser = argparse.ArgumentParser()
parser.add_argument('--glob', required=True, help='checkpoint pattern relative to checkpoints/')
parser.add_argument('--results', default='fidelity_results.csv')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class EncoderWithLinear(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        if head == 'linear':
            self.linear_classifier = nn.Linear(2048, 10)
        elif head == 'MLP':
            self.linear_classifier = nn.Sequential(
                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(512, 10),
            )
        else:  # 'proposed': 4-layer head of table2_proposed.py
            self.linear_classifier = nn.Sequential(
                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 10),
            )

    def forward(self, x):
        return self.linear_classifier(self.encoder(x, out='h'))


def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            preds.append(model(inputs.to(device)).argmax(1).cpu())
            labels.append(targets)
    return torch.cat(preds), torch.cat(labels)


# victim: WideResNet-40x10 stored as a whole module in the .t7 checkpoint
victim = torch.load(data_path('wide-resnet-40x10.t7'), map_location=device, weights_only=False)['net'].to(device)

victim_cache = {}  # budget -> (victim predictions, true labels, loader)


def victim_on(budget):
    if budget not in victim_cache:
        with open(data_path(f'random_{budget}_test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
        loader = DataLoader(TransformedDataset(test_dataset, transform_test), batch_size=256, shuffle=False, num_workers=2)
        vp, labels = predict(victim, loader)
        print(f'victim accuracy on {budget} test split ({len(labels)} samples): {100.0 * (vp == labels).float().mean():.2f}%')
        victim_cache[budget] = (vp, labels, loader)
    return victim_cache[budget]


name_re = re.compile(r'(?P<budget>4k|2k|1k|250)_(?P<model>linear|MLP)_(?P<method>finetune|layer4|frozen)_seed(?P<seed>\d+)(?P<final>_final)?\.pth$')
# Table 2 proposed-method checkpoints: {budget}_conf{c}_sim{s}_seed{seed}[_final].pth
name_re2 = re.compile(r'(?P<budget>4k|2k|1k|250)_conf(?P<conf>[0-9.]+)_sim(?P<sim>[0-9.]+)_seed(?P<seed>\d+)(?P<final>_final)?\.pth$')

results_file = out_path(args.results)
write_header = not os.path.exists(results_file)
with open(results_file, 'a', newline='') as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(['checkpoint', 'budget', 'model', 'training_method', 'seed', 'selection',
                    'accuracy', 'fidelity', 'victim_accuracy', 'n_test'])
    for path in sorted(glob.glob(ckpt_path(args.glob))):
        m = name_re.search(os.path.basename(path))
        if m:
            head, method = m['model'], m['method']
        else:
            m = name_re2.search(os.path.basename(path))
            if not m:
                print('skip (unrecognised name):', path)
                continue
            head, method = 'proposed', f"proposed_conf{m['conf']}_sim{m['sim']}"
        vp, labels, loader = victim_on(m['budget'])
        encoder = EncodeProject()
        encoder.fc = nn.Identity()
        model = EncoderWithLinear(encoder, head).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        ep, _ = predict(model, loader)
        acc = 100.0 * (ep == labels).float().mean().item()
        fid = 100.0 * (ep == vp).float().mean().item()
        vacc = 100.0 * (vp == labels).float().mean().item()
        sel = 'final' if m['final'] else 'best'
        print(f"{os.path.basename(path)}: acc {acc:.2f}  fidelity {fid:.2f}")
        w.writerow([os.path.relpath(path, ckpt_path('')), m['budget'], head, method, m['seed'], sel,
                    f'{acc:.2f}', f'{fid:.2f}', f'{vacc:.2f}', len(labels)])
