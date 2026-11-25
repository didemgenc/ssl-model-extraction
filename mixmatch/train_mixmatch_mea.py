"""MixMatch model-extraction baseline (Jagielski et al. 2020 setting) against OUR victim.

Default --setup original (Table 3): labeled = class-balanced random queries from the CIFAR-10 TRAINING set
(the same query sets as our Table 3 experiments) with the victim's labels; unlabeled = the 50k training set;
test = the full 10k CIFAR-10 test set (victim accuracy 94.40%). --setup split uses the Table 1-2 data instead.

Uses the WRN-28-2 model and the MixMatch training step of the unofficial PyTorch implementation
(YU1ut/MixMatch-pytorch, cloned into ../mixmatch_github, unmodified) with the default hyper-parameters
(Adam lr 0.002, batch 64, 1024 iterations per epoch, lambda_u 75, T 0.5, K 2, alpha 0.75, EMA 0.999).
Data (exported by cifar10/export_mixmatch_inputs.py):
  labeled   = our query set (CIFAR-10 test images) with the VICTIM's labels
  unlabeled = the full CIFAR-10 training set (50k), as in our method
  test      = our fixed 6000-image held-out split; no validation set, no model selection
Reports accuracy and fidelity (agreement with the victim) of the EMA model after the final epoch.
Checkpoint is saved every epoch and the run resumes automatically if restarted.

  python train_mixmatch_mea.py --budget 4k --epochs 128
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'mixmatch_github'))
import models.wideresnet as wrn            # noqa: E402  (MixMatch repo)
import dataset.cifar10 as mmdata           # noqa: E402  (MixMatch repo: transforms, normalize, transpose)

sys.path.insert(0, os.path.join(HERE, '..', 'cifar10'))
from paths import DATASET_ROOT             # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--budget', choices=['4k', '2k', '1k', '250'], required=True)
parser.add_argument('--epochs', default=128, type=int)
parser.add_argument('--train-iteration', default=1024, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--setup', default='original', choices=['original', 'split'],
                    help='original: queries from the CIFAR-10 training set, full 10k test set (Jagielski et al. setting, Table 3); '
                         'split: queries from the test set, 6000-image held-out split (Tables 1-2)')
parser.add_argument('--out', default=os.path.join(HERE, 'outputs'))
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
run_dir = os.path.join(args.out, f'{args.setup}_{args.budget}_ep{args.epochs}_seed{args.seed}')
os.makedirs(run_dir, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = 'cuda'

# ---------------- data ----------------
inp_name = f'mixmatch_inputs_{args.budget}.pt' if args.setup == 'split' else f'mixmatch_inputs_orig_{args.budget}.pt'
inp = torch.load(os.path.join(HERE, '..', 'cifar10', 'outputs', inp_name))
train_base = torchvision.datasets.CIFAR10(DATASET_ROOT, train=True, download=False)
test_base = torchvision.datasets.CIFAR10(DATASET_ROOT, train=False, download=False)


class ArrayDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = mmdata.transpose(mmdata.normalize(data))
        self.targets = np.asarray(targets)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.targets[i]


transform_train = transforms.Compose([mmdata.RandomPadandCrop(32), mmdata.RandomFlip(), mmdata.ToTensor()])
transform_val = transforms.Compose([mmdata.ToTensor()])

query_base = train_base if inp.get('query_source', 'test') == 'train' else test_base
labeled_set = ArrayDataset(query_base.data[inp['query_idx'].numpy()], inp['query_victim_label'].numpy(), transform_train)
unlabeled_set = ArrayDataset(train_base.data, -np.ones(len(train_base.data), dtype=np.int64), mmdata.TransformTwice(transform_train))
test_set = ArrayDataset(test_base.data[inp['test_idx'].numpy()], inp['test_true'].numpy(), transform_val)
victim_pred = inp['test_victim']
print(f'#Labeled (query, victim labels): {len(labeled_set)}  #Unlabeled: {len(unlabeled_set)}  #Test: {len(test_set)}  '
      f'victim acc on test split: {100.0 * (victim_pred == inp["test_true"]).float().mean():.2f}%')

labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)


# ---------------- model / MixMatch (same as the repo's train.py) ----------------
def create_model(ema=False):
    m = wrn.WideResNet(num_classes=10).to(device)
    if ema:
        for p in m.parameters():
            p.detach_()
    return m


model, ema_model = create_model(), create_model(ema=True)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


class WeightEMA:
    def __init__(self, model, ema_model, alpha):
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.alpha, self.wd = alpha, 0.02 * args.lr
        for p, e in zip(self.params, self.ema_params):
            p.data.copy_(e.data)

    def step(self):
        for p, e in zip(self.params, self.ema_params):
            if e.dtype == torch.float32:
                e.mul_(self.alpha).add_(p * (1 - self.alpha))
                p.mul_(1 - self.wd)


ema_opt = WeightEMA(model, ema_model, args.ema_decay)


def linear_rampup(current):
    return float(np.clip(current / args.epochs, 0.0, 1.0))


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train_epoch(epoch):
    model.train()
    li, ui = iter(labeled_loader), iter(unlabeled_loader)
    tot = 0.0
    for b in range(args.train_iteration):
        try:
            x, y = next(li)
        except StopIteration:
            li = iter(labeled_loader); x, y = next(li)
        try:
            (u1, u2), _ = next(ui)
        except StopIteration:
            ui = iter(unlabeled_loader); (u1, u2), _ = next(ui)
        bs = x.size(0)
        y = torch.zeros(bs, 10).scatter_(1, y.view(-1, 1).long(), 1)
        x, y, u1, u2 = x.to(device), y.to(device), u1.to(device), u2.to(device)
        with torch.no_grad():
            p = (torch.softmax(model(u1), 1) + torch.softmax(model(u2), 1)) / 2
            pt = p ** (1 / args.T)
            tu = (pt / pt.sum(1, keepdim=True)).detach()
        all_x = torch.cat([x, u1, u2]); all_t = torch.cat([y, tu, tu])
        l = np.random.beta(args.alpha, args.alpha); l = max(l, 1 - l)
        idx = torch.randperm(all_x.size(0), device=device)
        mx = l * all_x + (1 - l) * all_x[idx]; mt = l * all_t + (1 - l) * all_t[idx]
        mx = interleave(list(torch.split(mx, bs)), bs)
        logits = [model(mx[0])] + [model(m) for m in mx[1:]]
        logits = interleave(logits, bs)
        lx = -torch.mean(torch.sum(F.log_softmax(logits[0], 1) * mt[:bs], 1))
        lu = torch.mean((torch.softmax(torch.cat(logits[1:]), 1) - mt[bs:]) ** 2)
        w = args.lambda_u * linear_rampup(epoch + b / args.train_iteration)
        loss = lx + w * lu
        optimizer.zero_grad(); loss.backward(); optimizer.step(); ema_opt.step()
        tot += loss.item()
    return tot / args.train_iteration


def evaluate(m):
    m.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            preds.append(m(x.to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    return 100.0 * (preds == inp['test_true']).float().mean().item(), 100.0 * (preds == victim_pred).float().mean().item()


# ---------------- resume ----------------
ckpt_file = os.path.join(run_dir, 'checkpoint.pth.tar')
start = 0
if os.path.exists(ckpt_file):
    ck = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ck['state_dict']); ema_model.load_state_dict(ck['ema_state_dict'])
    optimizer.load_state_dict(ck['optimizer']); start = ck['epoch']
    torch.set_rng_state(ck['rng']); np.random.set_state(ck['np_rng']); random.setstate(ck['py_rng'])
    print(f'resumed from epoch {start}')

log = open(os.path.join(run_dir, 'log.csv'), 'a')
if start == 0:
    log.write('epoch,train_loss,test_acc,fidelity,minutes\n')
t0 = time.time()
for epoch in range(start, args.epochs):
    loss = train_epoch(epoch)
    acc, fid = evaluate(ema_model)
    print(f'epoch {epoch + 1}/{args.epochs} loss {loss:.4f} test acc {acc:.2f} fidelity {fid:.2f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    log.write(f'{epoch + 1},{loss:.4f},{acc:.2f},{fid:.2f},{(time.time() - t0) / 60:.1f}\n'); log.flush()
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(), 'np_rng': np.random.get_state(),
                'py_rng': random.getstate()}, ckpt_file)
log.close()

acc, fid = evaluate(ema_model)
torch.save(ema_model.state_dict(), os.path.join(run_dir, 'ema_final.pth'))
print(f'RESULT mixmatch setup={args.setup} budget={args.budget} epochs={args.epochs} seed={args.seed} final_acc={acc:.2f} fidelity={fid:.2f}')
res = os.path.join(args.out, 'mixmatch_results.csv')
new = not os.path.exists(res)
with open(res, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(['setup', 'budget', 'epochs', 'iterations_per_epoch', 'seed', 'final_acc', 'fidelity', 'victim_acc', 'n_labeled', 'n_unlabeled', 'n_test'])
    w.writerow([args.setup, args.budget, args.epochs, args.train_iteration, args.seed, f'{acc:.2f}', f'{fid:.2f}',
                f'{100.0 * (victim_pred == inp["test_true"]).float().mean():.2f}', len(labeled_set), len(unlabeled_set), len(test_set)])
