# -*- coding: utf-8 -*-
"""Контроль відтворюваності статті 1: прогони s1_verify (HEDAC*, хмара) проти архіву ПК conf_main.
Порівнюються моменти виявлення кожного постраждалого (побітово) і SWS (абсолютна різниця)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
a = {}
for l in open(os.path.join(R, 'conf_main', 'runs.jsonl'), encoding='utf-8'):
    r = json.loads(l)
    if r['method'] == 'HEDAC*':
        a[r['instance']] = r
n = same_t = 0
dmax = 0.0
for l in open(os.path.join(R, 's1_verify', 'runs.jsonl'), encoding='utf-8'):
    r = json.loads(l)
    b = a[r['instance']]
    n += 1
    same_t += r['found_t'] == b['found_t'] and r['victims'] == b['victims']
    dmax = max(dmax, abs(r['sws'] - b['sws']))
out = dict(n=n, found_t_identical=same_t, max_abs_sws_diff=dmax)
json.dump(out, open(os.path.join(R, 's1_verify', 'verify_vs_archive.json'), 'w'), indent=1)
print(out)
sys.exit(0 if same_t == n and dmax < 1e-12 else 1)
