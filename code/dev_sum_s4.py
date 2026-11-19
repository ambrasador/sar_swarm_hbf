# -*- coding: utf-8 -*-
"""Підсумок розроблення S4: J̄ (середнє за 6 умовами) і по умовах. python dev_sum_s4.py <exp> [<exp2> ...]"""
import json, sys, os, collections
import numpy as np
CONDS = ['A0', 'A40', 'A80', 'B0', 'B40', 'B80']
def load(exps):
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in exps:
        for line in open(os.path.join('results', e, 'runs.jsonl'), encoding='utf-8'):
            if line.strip():
                r = json.loads(line); R[r['method']][r['cond']].append(r['sws'])
    return R
def table(exps, minn=1):
    R = load(exps)
    rows = []
    for m, d in R.items():
        if all(len(d[c]) >= minn for c in CONDS):
            per = [float(np.mean(d[c])) for c in CONDS]
            rows.append((float(np.mean(per)), m, per, len(d['A0'])))
    rows.sort(reverse=True)
    return rows
if __name__ == '__main__':
    rows = table(sys.argv[1:])
    print('метод'.ljust(24), 'J̄    ', '  '.join(c.ljust(5) for c in CONDS), ' n/умову')
    for j, m, per, n in rows:
        print(m.ljust(24), f'{j:.4f}', '  '.join(f'{x:.3f}' for x in per), ' ', n)
