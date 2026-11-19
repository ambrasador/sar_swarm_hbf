# -*- coding: utf-8 -*-
"""Рисунок 4 статті 1: різниця SWS у підтверджувальних умовах S4 (HBF-S − HEDAC*-ε*; HBF-S − HBF-S-fix; ORACLE-S − HEDAC*-ε*).
Запуск: python make_fig4.py <analysis_s4_main.json> <тека виходу>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A, OUT = sys.argv[1:3]
plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.6})
a = json.load(open(A, encoding='utf-8'))
W = 8 / 2.54
conds = ['A0', 'A40', 'A80', 'B0', 'B40', 'B80']
labs = ['A, 0', 'A, 0,4', 'A, 0,8', 'B, 0', 'B, 0,4', 'B, 0,8']
x = np.arange(6)
fig, ax = plt.subplots(figsize=(W, W * 0.78))
for key, lab, off, st in (('HBFS|EPS', 'HBF-S − HEDAC*-ε*', -0.14, dict(color='black', marker='o')),
                          ('HBFS|FIX', 'HBF-S − HBF-S-fix', 0.14, dict(color='0.45', marker='s', mfc='white'))):
    d = np.array([a['cond'][key][c]['diff'] for c in conds])
    lo = np.array([a['cond'][key][c]['ci'][0] for c in conds]); hi = np.array([a['cond'][key][c]['ci'][1] for c in conds])
    ax.errorbar(x + off, d, yerr=[d - lo, hi - d], fmt=st['marker'], ms=3.5, capsize=1.5, lw=0.7, label=lab,
                color=st['color'], mfc=st.get('mfc', st['color']))
orc = np.array([a['means'][c]['ORACLE'] - a['means'][c]['EPS'] for c in conds])
ax.plot(x, orc, 'x', color='0.2', ms=4, mew=0.9, label='ORACLE-S − HEDAC*-ε* (запас)')
ax.axhline(0, color='0.3', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_xlabel('Вид помилки карти і частка ρ')
ax.set_ylabel('Різниця SWS')
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ',')))
ax.set_ylim(-0.055, 0.215)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, 'fig4_s4.png'), dpi=300)
fig.savefig(os.path.join(OUT, 'fig4_s4.tif'), dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
