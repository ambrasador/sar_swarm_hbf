# -*- coding: utf-8 -*-
"""
Стаття 1, розширення (листопад 2026): S4-M — істина ПОЗА класом гіпотез HBF-S (хибно задана модель помилки).
S4Map не змінюється; підклас лише підміняє alt_true. Гіпотези планувальника hyp_alts — ті самі, що в S4.
  M — вид A, зсув справжньої позиції d ~ U[22; 30] клітинок (гіпотези: 14 клітинок; найближча гіпотеза далі ніж 8).
  N — вид B, справжній розподіл — гаусіан σ = 3 у випадковій клітинці області [5, 44]^2, не збігається з жодною з 4 компонент.
Умови: M40, M80, N40, N80 (число — частка ρ у відсотках).
"""
import numpy as np
import s4_env as S4

VOFF = {'M': 500000, 'N': 600000}
BASE = {'M': 'A', 'N': 'B'}


class S4MMap(S4.S4Map):
    def __init__(self, kind, seed):
        self.kind_m = kind
        super().__init__(BASE[kind], seed)
        rng = np.random.default_rng(777_000 + seed * 13 + (0 if kind == 'M' else 1))
        if kind == 'M':
            r, c = self.lkp_rc
            d, th = rng.uniform(22.0, 30.0), rng.uniform(0.0, 2 * np.pi)
            self.alt_true = self._gaussian_at(r + d * np.sin(th), c + d * np.cos(th), S4.SIGMA_A)
            self.alt_info = dict(dist=float(d), theta=float(th))
        else:
            r, c = int(rng.integers(5, 45)), int(rng.integers(5, 45))
            self.alt_true = self._gaussian_at(r, c, 3.0)
            self.alt_info = dict(blob=[r, c])
