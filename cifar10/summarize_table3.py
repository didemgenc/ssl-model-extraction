"""Build the revised Table 3: MixMatch-based extraction (Jagielski et al. setup, re-run on our victim)
vs. the proposed method, single seed, queries drawn from the CIFAR-10 training set, full 10k test set.
Inputs : ../mixmatch/outputs/mixmatch_results.csv (setup=original rows)
         outputs/table3_results.csv (proposed method, sim 0.8, conf 0.9/0.95/0.95)
Outputs: outputs/table3_summary.csv, outputs/Table3_revised.tsv, outputs/Table3_revised.docx
"""
import csv
from paths import out_path

VICTIM = 94.40            # victim accuracy reported in the paper (full CIFAR-10 test set)
BUDGETS = ['4k', '1k', '250']
LABEL = {'4k': '4000', '1k': '1000', '250': '250'}

mm = {r['budget']: r for r in csv.DictReader(open('../mixmatch/outputs/mixmatch_results.csv'))
      if r['setup'] == 'original' and r['epochs'] == '128'}
pr = {r['budget']: r for r in csv.DictReader(open(out_path('table3_results.csv'))) if r['sim'] == '0.8'}

rows = []
for name, src, acc_key, fid_key in [('MixMatch-based SSL [6]', mm, 'final_acc', 'fidelity'),
                                    ('Proposed Method', pr, 'final_test_acc', 'final_fidelity')]:
    for b in BUDGETS:
        a, f = float(src[b][acc_key]), float(src[b][fid_key])
        rows.append([name, LABEL[b], f'{a:.2f}', f'{f:.2f}', f'{VICTIM:.2f}', f'{VICTIM - a:.2f}'])

header = ['Method', 'Query Budget', 'Extracted Model Test Accuracy (%)', 'Fidelity (%)',
          'Victim Model Test Accuracy (%)', 'Absolute Difference (%)']
with open(out_path('table3_summary.csv'), 'w', newline='') as fh:
    csv.writer(fh).writerows([header] + rows)
with open(out_path('Table3_revised.tsv'), 'w') as fh:
    fh.write('\t'.join(header) + '\n')
    for r in rows:
        fh.write('\t'.join(r) + '\n')
for r in [header] + rows:
    print('\t'.join(r))

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
except ImportError:
    raise SystemExit('python-docx not installed; CSV/TSV written only')

doc = Document()
t = doc.add_table(rows=1 + len(rows), cols=len(header))
t.style = 'Table Grid'
for j, h in enumerate(header):
    t.cell(0, j).text = h
for i, r in enumerate(rows, start=1):
    for j, v in enumerate(r):
        t.cell(i, j).text = v + ('%' if j >= 2 else '')
t.cell(1, 4).merge(t.cell(len(rows), 4)).text = f'{VICTIM:.2f}%'
for k in range(0, len(rows), len(BUDGETS)):
    t.cell(1 + k, 0).merge(t.cell(k + len(BUDGETS), 0)).text = rows[k][0]
for row in t.rows:
    for c in row.cells:
        for p in c.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.size = Pt(8)
doc.save(out_path('Table3_revised.docx'))
print('wrote', out_path('Table3_revised.docx'))
