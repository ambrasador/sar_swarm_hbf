# -*- coding: utf-8 -*-
"""Рисунки 2–3 статті 1 з результатів аналізу (analysis_s1.json) і телеметрії (results/s1_trace/trace.json).
Ширина — одна колонка (8 см), шрифт із метрикою Times New Roman (Liberation Serif), 300 dpi; TIFF (LZW) і PNG.
Запуск: python make_figs.py <analysis_s1.json> <trace.json> <тека виходу>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A, TR, OUT = sys.argv[1:4]
plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8, 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
res = json.load(open(A, encoding='utf-8'))
eps = res['frozen']['eps_star']
W = 8 / 2.54


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.tif'), dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


# ---- Рисунок 2: SWS залежно від частки постраждалих поза моделлю
RH = ['00', '20', '40', '60', '80']
x = np.array([0, .2, .4, .6, .8])
style = [('HEDAC*', 'HEDAC* (ε = 0,05)', dict(color='0.55', ls=':', marker='v')),
         ('BSS', 'BSS', dict(color='0.55', ls='--', marker='s')),
         ('HEDAC*-eps', f'HEDAC*-ε* (ε* = {str(eps).replace(".", ",")})', dict(color='0.25', ls='-.', marker='^')),
         ('HBF-fix', 'HBF-fix (β = 0)', dict(color='0.25', ls='--', marker='o', mfc='white')),
         ('HBF', 'HBF', dict(color='black', ls='-', marker='o'))]
fig, ax = plt.subplots(figsize=(W, W * 0.78))
for k, lab, st in style:
    m = np.array([res['means'][f'R{r}'][k]['mean'] for r in RH])
    lo = np.array([res['means'][f'R{r}'][k]['ci'][0] for r in RH])
    hi = np.array([res['means'][f'R{r}'][k]['ci'][1] for r in RH])
    off = {'HEDAC*': -0.012, 'BSS': -0.006, 'HEDAC*-eps': 0.0, 'HBF-fix': 0.006, 'HBF': 0.012}[k]
    ax.errorbar(x + off, m, yerr=[m - lo, hi - m], lw=0.9, ms=3.2, capsize=1.5, elinewidth=0.5, label=lab, **st)
ax.set_xlabel('Частка постраждалих поза апріорною моделлю ρ')
ax.set_ylabel('SWS')
ax.set_xticks(x); ax.set_xticklabels(['0', '0,2', '0,4', '0,6', '0,8'])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ',')))
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.3)
save(fig, 'fig2_rho')

# ---- Рисунок 3: апостеріорна оцінка ρ̂(t)
tr = json.load(open(TR, encoding='utf-8'))
S = tr['summary']
fig, ax = plt.subplots(figsize=(W, W * 0.72))
t = np.arange(len(S['R00']['est_mean']))
lines = [('S1', 'S1 (ρ = 0)', dict(color='0.6', ls=':')), ('R00', 'S3, ρ = 0', dict(color='0.6', ls='-')),
         ('R20', 'S3, ρ = 0,2', dict(color='0.45', ls='--')), ('R40', 'S3, ρ = 0,4', dict(color='0.3', ls='-.')),
         ('R60', 'S3, ρ = 0,6', dict(color='0.15', ls='--')), ('R80', 'S3, ρ = 0,8', dict(color='black', ls='-'))]
for k, lab, st in lines:
    ax.plot(t, S[k]['est_mean'], lw=1.0, label=lab, **st)
ax.set_xlabel('Крок пошуку t')
ax.set_ylabel(r'Апостеріорна оцінка $\hat{\rho}(t)$')
ax.set_ylim(0.1, 0.66)
ax.set_xlim(0, t[-1])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left', columnspacing=1.0, handlelength=2.2)
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.3)
save(fig, 'fig3_rhohat')
print('ok')
