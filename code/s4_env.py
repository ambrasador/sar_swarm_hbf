# -*- coding: utf-8 -*-
"""
Стаття 1, розширення (листопад 2026): сценарії зі СТРУКТУРОВАНОЮ помилкою апріорної карти.
env.py не змінюється: модуль додає підклас MapInstance, який повторює генерацію карт S1 / S2 побітово
(перевіряє tests_s4.py) і додатково зберігає
  * alt_true   — справжній розподіл постраждалих «поза моделлю» (невідомий рою);
  * hyp_alts   — альтернативні карти-гіпотези, які планувальник МОЖЕ сформулювати без знання істини.

Види помилки:
  A — «зміщена остання відома позиція (ОВП)»: геометрія S1 (гаусіан σ = 6 навколо ОВП). Постраждалі поза моделлю
      розподілені навколо справжньої позиції, зміщеної від повідомленої ОВП на відстань d ~ U[10; 18] клітинок у
      випадковому напрямку. Гіпотези планувальника: гаусіани σ = 6 навколо ОВП, зміщеної на 14 клітинок у
      одному з 8 напрямків (кроки 45°).
  B — «хибна категорія поведінки»: геометрія S2 (мультимодальна карта з чотирьох компонент: околиця точки, стежка,
      річка, споруда). Постраждалі поза моделлю дотримуються однієї компоненти (випадкової, j*), а не суміші.
      Гіпотези планувальника: кожна з чотирьох компонент окремо.

Умови підтвердження: A0, A40, A80, B0, B40, B80 (літера — вид, число — частка ρ постраждалих поза моделлю, %).
"""
import numpy as np

import env as E
from env import MapInstance, N, W, H, RR, CC, K_VICTIMS

KINDS = {'A': 'S1', 'B': 'S2'}
VOFF = {'A': 200000, 'B': 300000}       # зміщення зерна генератора постраждалих
SIGMA_A = 6.0
HYP_SHIFT = 14.0


class S4Map(MapInstance):
    def __init__(self, kind, seed, lam_len=5.0):
        self.kind = kind
        super().__init__(KINDS[kind], seed, lam_len)
        self._gen_alts(seed)

    # --- апріор: дослівно як у MapInstance._gen_prior, із збереженням компонент
    def _gen_prior(self, rng):
        g = np.zeros((H, W))
        R2, C2 = RR.reshape(H, W), CC.reshape(H, W)

        def gauss(r, c, s, w):
            return w * np.exp(-((R2 - r) ** 2 + (C2 - c) ** 2) / (2 * s * s))

        comps = []
        if self.scenario == 'S1':
            r, c = rng.integers(8, 28), rng.integers(10, 40)
            g += gauss(r, c, 6.0, 1.0)
            self.lkp_rc = (int(r), int(c))
        else:
            r, c = rng.integers(8, 30), rng.integers(8, 42)
            t0 = gauss(r, c, 4.0, 1.0) / (2 * np.pi * 16)
            g += t0
            comps.append(t0)
            r1, c1, r2, c2 = rng.integers(3, 35), rng.integers(0, 10), rng.integers(3, 35), rng.integers(40, 50)
            tt = np.linspace(0, 1, 60)
            line = np.zeros((H, W))
            for a in tt:
                line += gauss(r1 + a * (r2 - r1), c1 + a * (c2 - c1), 1.8, 1.0)
            t1 = 1.3 * line / line.sum()
            g += t1
            comps.append(t1)
            rr0, amp, ph = rng.integers(10, 35), rng.uniform(2, 6), rng.uniform(0, 6.28)
            river_r = rr0 + amp * np.sin(np.linspace(0, 2 * np.pi, W) + ph)
            band = np.exp(-(R2 - river_r[None, :]) ** 2 / (2 * 1.5 ** 2))
            t2 = 1.0 * band / band.sum()
            g += t2
            comps.append(t2)
            r, c = rng.integers(5, 38), rng.integers(5, 45)
            s = gauss(r, c, 2.2, 1.0)
            t3 = 0.8 * s / s.sum()
            g += t3
            comps.append(t3)
        self._comps_raw = comps
        g = g.ravel() * self.free
        self.prior = g / g.sum()

    def _norm(self, a):
        a = np.asarray(a, float).ravel() * self.free
        s = a.sum()
        if s < 1e-12:
            a = self.free.astype(float)
            s = a.sum()
        return a / s

    def _gaussian_at(self, r, c, sg):
        R2, C2 = RR.reshape(H, W), CC.reshape(H, W)
        r = float(np.clip(r, 0, H - 1)); c = float(np.clip(c, 0, W - 1))
        return self._norm(np.exp(-((R2 - r) ** 2 + (C2 - c) ** 2) / (2 * sg * sg)))

    def _gen_alts(self, seed):
        rng = np.random.default_rng(555_000 + seed * 13 + (0 if self.kind == 'A' else 1))
        if self.kind == 'A':
            r, c = self.lkp_rc
            d, th = rng.uniform(10.0, 18.0), rng.uniform(0.0, 2 * np.pi)
            self.alt_true = self._gaussian_at(r + d * np.sin(th), c + d * np.cos(th), SIGMA_A)
            self.hyp_alts = [self._gaussian_at(r + HYP_SHIFT * np.sin(k * np.pi / 4), c + HYP_SHIFT * np.cos(k * np.pi / 4), SIGMA_A)
                             for k in range(8)]
            self.alt_info = dict(dist=float(d), theta=float(th))
        else:
            comps = [self._norm(t) for t in self._comps_raw]
            j = int(rng.integers(4))
            self.alt_true = comps[j]
            self.hyp_alts = comps
            self.alt_info = dict(component=j)

    def sample_victims4(self, rng, rho, k=K_VICTIMS):
        """rho = 0 — як у S1/S2 (без повторень); інакше кожен постраждалий незалежно: з апріору (1 − rho) або з alt_true."""
        if rho <= 0:
            return self.sample_victims(rng, k), np.ones(k, int)
        out, flags = [], []
        for _ in range(k):
            if rng.random() < 1.0 - rho:
                out.append(rng.choice(N, p=self.prior)); flags.append(1)
            else:
                out.append(rng.choice(N, p=self.alt_true)); flags.append(0)
        return np.array(out), np.array(flags, int)
