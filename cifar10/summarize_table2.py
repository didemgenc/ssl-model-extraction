"""Summarise the CIFAR-10 rows of Table 2 (baseline = Table 1 FT/linear, proposed = table2_proposed.py).

Reads  outputs/fidelity_results.csv (Table 1) and outputs/fidelity_table2.csv (proposed method),
keeps the final-epoch checkpoints, and writes mean ± std over seeds to
  outputs/table2_summary_final.csv, outputs/Table2_revised.tsv, outputs/Table2_revised.docx
Victim accuracy is the single value on the full 10k CIFAR-10 test set (94.40%).
"""
import csv
import statistics as st
from collections import defaultdict
from paths import out_path

VICTIM = 94.40
BUDGETS = ['4k', '2k', '1k', '250']
LABEL = {'4k': '4000', '2k': '2000', '1k': '1000', '250': '250'}


def load(path, keep):
    acc, fid = defaultdict(list), defaultdict(list)
    with open(out_path(path)) as f:
        for r in csv.DictReader(f):
            if r['selection'] != 'final' or not keep(r):
                continue
            acc[r['budget']].append(float(r['accuracy']))
            fid[r['budget']].append(float(r['fidelity']))
    return acc, fid


def ms(v):
    return f'{st.mean(v):.2f} ± {st.stdev(v):.2f}' if len(v) > 1 else f'{v[0]:.2f}'


rows = []
base = load('fidelity_results.csv', lambda r: r['model'] == 'linear' and r['training_method'] == 'finetune')
prop = load('fidelity_table2.csv', lambda r: r['model'] == 'proposed')
for name, (acc, fid) in [('Baseline Method', base), ('Proposed Method', prop)]:
    for b in BUDGETS:
        rows.append([name, LABEL[b], len(acc[b]), ms(acc[b]), ms(fid[b]), f'{VICTIM:.2f}', f'{VICTIM - st.mean(acc[b]):.2f}'])

hdr = ['Method', 'Query Budget', 'n_seeds', 'Extracted Model Test Accuracy (%)', 'Fidelity (%)',
       'Victim Model Test Accuracy (%)', 'Absolute Difference (%)']
with open(out_path('table2_summary_final.csv'), 'w', newline='') as f:
    csv.writer(f).writerows([hdr] + rows)
with open(out_path('Table2_revised.tsv'), 'w') as f:
    f.write('\t'.join(hdr) + '\n')
    for r in rows:
        f.write('\t'.join(map(str, r)) + '\n')
for r in rows:
    print(r)

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
except ImportError:
    raise SystemExit('python-docx not installed; CSV/TSV written only')

doc = Document()
cols = ['Dataset', 'Method', 'Query Budget', 'Extracted Model Test Accuracy', 'Fidelity',
        'Victim Model Test Accuracy', 'Absolute Difference']
t = doc.add_table(rows=1 + len(rows), cols=len(cols))
t.style = 'Table Grid'
for j, c in enumerate(cols):
    t.cell(0, j).text = c
for i, r in enumerate(rows, start=1):
    vals = ['CIFAR-10', r[0], r[1], r[3] + '%', r[4] + '%', r[5] + '%', r[6] + '%']
    for j, v in enumerate(vals):
        t.cell(i, j).text = v
# merge Dataset column, Method blocks, victim column
t.cell(1, 0).merge(t.cell(len(rows), 0)).text = 'CIFAR-10'
t.cell(1, 5).merge(t.cell(len(rows), 5)).text = f'{VICTIM:.2f}%'
for k in range(0, len(rows), len(BUDGETS)):
    t.cell(1 + k, 1).merge(t.cell(k + len(BUDGETS), 1)).text = rows[k][0]
for row in t.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.size = Pt(8)
doc.save(out_path('Table2_revised.docx'))
print('wrote', out_path('Table2_revised.docx'))
