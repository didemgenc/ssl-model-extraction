"""Summarise Table 1 runs as mean ± std over seeds (Jagielski-style multi-trial reporting).

Reads outputs/table1_results.csv (from table1_baseline.py) and, if present,
outputs/fidelity_results.csv (from eval_fidelity.py). Prints a markdown table and writes
outputs/table1_summary.csv.

  python summarize_table1.py            # final-epoch accuracy (no test-set model selection)
  python summarize_table1.py --best     # best-epoch accuracy (what the original script tracked)
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

from paths import out_path

parser = argparse.ArgumentParser()
parser.add_argument('--best', action='store_true', help='summarise best-epoch instead of final-epoch accuracy')
args = parser.parse_args()
col = 'best_test_acc' if args.best else 'final_test_acc'
sel = 'best' if args.best else 'final'

BUDGETS = ['4k', '2k', '1k', '250']
METHODS = [('frozen', 'F'), ('layer4', 'LL'), ('finetune', 'FT')]
MODELS = ['linear', 'MLP']

acc = defaultdict(list)
for r in csv.DictReader(open(out_path('table1_results.csv'))):
    acc[(r['budget'], r['training_method'], r['model'])].append(float(r[col]))

fid = defaultdict(list)
fid_file = out_path('fidelity_results.csv')
if os.path.exists(fid_file):
    for r in csv.DictReader(open(fid_file)):
        if r['selection'] == sel:
            fid[(r['budget'], r['training_method'], r['model'])].append(float(r['fidelity']))


def ms(vals):
    if not vals:
        return '-'
    if len(vals) == 1:
        return f'{vals[0]:.2f}'
    return f'{st.mean(vals):.2f} ± {st.stdev(vals):.2f}'


n_seeds = {len(v) for v in acc.values()}
print(f'accuracy column: {col}; seeds per cell: {sorted(n_seeds)}; fidelity rows: {sum(len(v) for v in fid.values())}\n')
header = '| Budget | Method | Linear acc | Linear fid | MLP acc | MLP fid |'
print(header)
print('|---|---|---|---|---|---|')
rows = []
for b in BUDGETS:
    for m, lab in METHODS:
        cells = []
        for model in MODELS:
            k = (b, m, model)
            cells += [ms(acc[k]), ms(fid[k])]
            rows.append([b, lab, model, len(acc[k]), ms(acc[k]), ms(fid[k])])
        print(f'| {b} | {lab} | ' + ' | '.join(cells) + ' |')

with open(out_path('table1_summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['budget', 'method', 'model', 'n_seeds', f'accuracy_{sel}', f'fidelity_{sel}'])
    w.writerows(rows)
