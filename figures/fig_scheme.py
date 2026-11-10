# -*- coding: utf-8 -*-
"""Рис. 1 статті 1: схема методу HBF (вертикальна блок-схема на ширину колонки 8 см)."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8, 'mathtext.fontset': 'stix'})
OUT = sys.argv[1] if len(sys.argv) > 1 else 'fig1_scheme'
W, H = 8 / 2.54, 9.35 / 2.54
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 80); ax.set_ylim(0, 93.6); ax.axis('off')

boxes = [
    (87, 'Апріорна карта $\\psi$ і гіпотези $\\rho_1<\\ldots<\\rho_K$\nз апріорними вагами $\\pi_k^0$'),
    (73, '$K$ карт переконань $q^{(k)}$: початок за (6) з $\\varepsilon=\\rho_k$,\nоновлення кожним оглядом за (7)'),
    (59, 'Правдоподібність історії оглядів $\\ell_k(t)$ (10)\n→ апостеріорні ваги $\\pi_k(t)$ (11)'),
    (45, 'Апостеріорно-предиктивна карта\n$\\bar q(t)=\\Sigma_k\\,\\pi_k(t)\\,q^{(k)}(t)$ (12)'),
    (31, 'Потенціальне поле $\\varphi=\\mathbf{G}(\\bar q\\odot \\bar p)$ (8)'),
    (17.5, 'Мікрокрок агентів: максимум $\\varphi$ серед сусідів,\nінерція, заборона сусідства обраних клітинок'),
]
bw, bh, x0 = 62, 10, 6
for y, t in boxes:
    ax.add_patch(FancyBboxPatch((x0, y - bh / 2), bw, bh, boxstyle='round,pad=0.4,rounding_size=1.5',
                                fc='white', ec='black', lw=0.7))
    ax.text(x0 + bw / 2, y, t, ha='center', va='center', linespacing=1.25)
for (y1, _), (y2, _) in zip(boxes[:-1], boxes[1:]):
    ax.add_patch(FancyArrowPatch((x0 + bw / 2, y1 - bh / 2 - 0.4), (x0 + bw / 2, y2 + bh / 2 + 0.4),
                                 arrowstyle='-|>', mutation_scale=7, lw=0.7, color='black'))
# рій і зворотний зв'язок спостережень
ax.add_patch(FancyBboxPatch((x0, 2.2), bw, 6.5, boxstyle='round,pad=0.4,rounding_size=1.5', fc='#eeeeee', ec='black', lw=0.7))
ax.text(x0 + bw / 2, 5.45, 'Рій із $M$ агентів: огляди сліду $F(x)$, виявлення', ha='center', va='center')
ax.add_patch(FancyArrowPatch((x0 + bw / 2, 17.5 - bh / 2 - 0.4), (x0 + bw / 2, 9.3), arrowstyle='-|>', mutation_scale=7, lw=0.7, color='black'))
xr = x0 + bw + 2.5
ax.plot([x0 + bw + 0.4, xr + 3, xr + 3], [5.45, 5.45, 73], color='black', lw=0.7)
ax.add_patch(FancyArrowPatch((xr + 3, 73), (x0 + bw + 0.6, 73), arrowstyle='-|>', mutation_scale=7, lw=0.7, color='black'))
ax.text(xr + 4.3, 38, 'спостереження кроку $t$', rotation=90, ha='center', va='center', fontsize=7)
fig.savefig(OUT + '.png', dpi=300)
fig.savefig(OUT + '.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
print('ok')
