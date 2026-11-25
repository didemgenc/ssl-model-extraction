"""Export the query set (victim-labelled) and held-out test split of each budget as plain tensors,
plus the victim's predictions on the test split, so that the MixMatch baseline can use exactly the
same data without importing this project's pickles. Output: outputs/mixmatch_inputs_{budget}.pt
"""
from paths import data_path, out_path, orig_path, DATASET_ROOT
import argparse
import pickle
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models.wideresnet import Wide_ResNet  # noqa: F401


class ModifiedLabelsDataset(Dataset):
    def __init__(self, original_dataset, target_labels_dict):
        self.original_dataset, self.target_labels_dict = original_dataset, target_labels_dict
        self.filtered_indices = [idx for idx in range(len(original_dataset)) if idx in target_labels_dict]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        i = self.filtered_indices[index]
        image, _ = self.original_dataset[i]
        return image, self.target_labels_dict.get(i, -1), i


class CustomDataset(Dataset):
    def __init__(self, trainset, indexes, transform=None):
        self.trainset, self.indexes, self.transform = trainset, indexes, transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        i = self.indexes[idx]
        image, label = self.trainset[i]
        if self.transform:
            image = self.transform(image)
        return image, label, i


parser = argparse.ArgumentParser()
parser.add_argument('--setup', default='split', choices=['split', 'original'],
                    help="split: queries from the CIFAR-10 test set, 6000-image held-out test split (Tables 1-2); "
                         "original: queries from the CIFAR-10 training set, full 10k test set (Table 3 / MixMatch setting)")
args = parser.parse_args()
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
device = 'cuda'
victim = torch.load(data_path('wide-resnet-40x10.t7'), map_location=device, weights_only=False)['net'].to(device).eval()
for budget in ['4k', '2k', '1k', '250']:
    if args.setup == 'split':
        with open(data_path(f'{budget}_target_labeled_dataset.pkl'), 'rb') as f:
            q = pickle.load(f)
        with open(data_path(f'random_{budget}_test_dataset.pkl'), 'rb') as f:
            t = pickle.load(f)
        out_name, q_src = f'mixmatch_inputs_{budget}.pt', 'test'
    else:
        with open(orig_path(f'random_{budget}_target_labeled_dataset.pkl'), 'rb') as f:
            q = pickle.load(f)
        test_base = torchvision.datasets.CIFAR10(DATASET_ROOT, train=False, download=False)
        t = CustomDataset(test_base, list(range(len(test_base))))
        out_name, q_src = f'mixmatch_inputs_orig_{budget}.pt', 'train'
    q_idx = torch.tensor(q.filtered_indices)
    q_lab = torch.tensor([q.target_labels_dict[i] for i in q.filtered_indices])
    t_idx = torch.tensor(list(t.indexes))
    loader = DataLoader(CustomDataset(t.trainset, t.indexes, tf), batch_size=256, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            preds.append(victim(x.to(device)).argmax(1).cpu()); trues.append(torch.as_tensor(y))
    preds, trues = torch.cat(preds), torch.cat(trues)
    if q_src == 'test':
        assert len(set(q_idx.tolist()) & set(t_idx.tolist())) == 0
    torch.save({'query_idx': q_idx, 'query_victim_label': q_lab, 'query_source': q_src,
                'test_idx': t_idx, 'test_true': trues, 'test_victim': preds}, out_path(out_name))
    print(f'{budget}: {len(q_idx)} query, {len(t_idx)} test, victim acc {100.0 * (preds == trues).float().mean():.2f}%, '
          f'query victim-label acc {100.0 * (q_lab == torch.tensor([q.original_dataset.targets[i] for i in q.filtered_indices])).float().mean():.2f}%')
