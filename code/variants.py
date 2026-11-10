# -*- coding: utf-8 -*-
"""
Варіанти методів для абляцій (перевірка гіпотез розділу 2) і класи з параметрами для
симетричного підбору (Е6 v2). Оригінальні класи (baselines.py, novel.py, hybrid.py) НЕ змінюються.

Кожен варіант = (клас, словник параметрів). Параметри встановлюються як атрибути екземпляра
перед reset(). За типових параметрів класи *T відтворюють базові методи побітово
(перевіряє tests/test_extensions.py).
"""
import numpy as np
import baselines as B
import novel as NV
import hybrid as HB
from env import N, W, H, RC, valid_nbrs, step_toward, snap, survival


# ------------------------------------------------------------------ SIFD без кривої виживання
class SIFDx(NV.SIFD):
    """SIFD з перемикачем множника S(t + d_ik) у корисності ділянки (surv_mode: 'on' | 'off')."""
    surv_mode = 'on'

    def allocate(self, env, V, tgt):
        mp, M = env.mp, env.M
        d = mp.apsp[np.ix_(env.pos, tgt)].astype(float)
        S = survival(env.t + d) if self.surv_mode == 'on' else 1.0
        base_u = V[None, :] * S / (d + self.h)
        base_u[:, V <= 0] = 0
        for _ in range(8):
            changed = False
            for i in self.rng.permutation(M):
                others = np.delete(self.assign, i)
                n = np.bincount(others[others >= 0], minlength=self.K)
                g = base_u[i] / (n + 1.0) ** self.m
                b = int(np.argmax(g))
                a = self.assign[i]
                if a < 0 or g[b] > (1 + self.hyst) * g[a]:
                    if b != a:
                        self.assign[i] = b
                        changed = True
            if not changed:
                break


# ------------------------------------------------------------------ BSS з фіксованими вагами моделей
class BSSFixed(NV.BSS):
    """BSS з фіксованою довірою: mode='prior' — лише модель M0 (без скептиків),
    mode='uniform' — лише рівномірна модель M1 (усі агенти — скептики)."""
    mode = 'prior'

    def weights(self):
        return np.array([1.0, 0.0]) if self.mode == 'prior' else np.array([0.0, 1.0])


# ------------------------------------------------------------------ класи з параметрами (Е6 v2)
class LawnT(B.Lawnmower):
    """LAWN з параметром кроку галсів lane (типово 3 = env.LANE)."""
    lane = 3

    def reset(self, env, rng):
        B.Method.reset(self, env, rng)
        mp, M = env.mp, env.M
        strips = np.linspace(0, W, M + 1).astype(int)
        self.wps, self.k = [], np.zeros(M, int)
        for i in range(M):
            a, b = strips[i], strips[i + 1]
            lanes = list(range(a + 1, b, int(self.lane)))
            if not lanes:
                lanes = [a]
            if lanes[-1] < b - 2:
                lanes.append(b - 2)
            pts = []
            for j, lc in enumerate(lanes):
                rows = (H - 1, 0) if j % 2 == 0 else (0, H - 1)
                pts += [snap(mp, (rows[0], lc)), snap(mp, (rows[1], lc))]
            self.wps.append(pts)


class LevyT(B.Levy):
    """LEVY з параметрами показника mu_exp (типово 2: pareto(1.0)) і обмеження довжини cap (25)."""
    mu_exp, cap = 2.0, 25

    def act(self, env):
        mp, out = env.mp, env.pos.copy()
        for i in range(env.M):
            if self.left[i] <= 0 or mp.nbr[env.pos[i], self.dir[i]] < 0:
                ok = np.nonzero(mp.nbr[env.pos[i]] >= 0)[0]
                self.dir[i] = self.rng.choice(ok)
                self.left[i] = int(min(int(self.cap), np.floor(self.rng.pareto(self.mu_exp - 1.0) + 1)))
            out[i] = mp.nbr[env.pos[i], self.dir[i]]
            self.left[i] -= 1
        return out


class GWOT(B.GWO):
    """GWO з параметром початкового значення a0 (типово 2: a = a0 (1 - t/T))."""
    a0 = 2.0

    def act(self, env):
        mp, F = env.mp, env.gain_map()
        for i in range(env.M):
            w = mp.win[env.pos[i]]
            self.known.update(w[w < N].tolist())
        cells = np.fromiter(self.known, int)
        leaders = B.top_distinct(cells, F[cells], 3)
        a = self.a0 * (1 - env.t / env.T)
        out = env.pos.copy()
        for i in range(env.M):
            X = RC[env.pos[i]]
            Xs = []
            for L in leaders:
                A = 2 * a * self.rng.random(2) - a
                C = 2 * self.rng.random(2)
                D = np.abs(C * RC[L] - X)
                Xs.append(RC[L] - A * D)
            tgt = snap(mp, np.mean(Xs, 0))
            out[i] = step_toward(mp, env.pos[i], tgt, self.rng) if tgt != env.pos[i] else \
                self.rng.choice(valid_nbrs(mp, env.pos[i]))
        return out


# класи, що використовуються для методу з підібраними параметрами (Е6 v2)
TUNABLE = {'LAWN': LawnT, 'LEVY': LevyT, 'GWO': GWOT}

# ------------------------------------------------------------------ реєстр варіантів
# назва -> (клас, параметри, базовий метод, що перевіряє)
VARIANTS = {
    # SIFD (гіпотези H1.3, H1.4)
    'SIFD-noS':   (SIFDx, {'surv_mode': 'off'}, 'SIFD', 'без множника виживання S(t+d) у корисності ділянки'),
    'SIFD-m0':    (NV.SIFD, {'m': 0.0}, 'SIFD', 'без інтерференції (m = 0)'),
    'SIFD-m2':    (NV.SIFD, {'m': 2.0}, 'SIFD', 'посилена інтерференція (m = 2)'),
    'SIFD-P5':    (NV.SIFD, {'P': 5}, 'SIFD', 'дрібні ділянки 5x5'),
    'SIFD-P16':   (NV.SIFD, {'P': 16}, 'SIFD', 'великі ділянки 16x16'),
    'SIFD-noHyst': (NV.SIFD, {'hyst': 0.0}, 'SIFD', 'без гістерезису перерозподілу'),
    # PAS (гіпотези H2.3, H2.4)
    'PAS-g0':     (NV.PAS, {'gamma': 0.0}, 'PAS', 'без обміну намірами (gamma = 0)'),
    'PAS-g05':    (NV.PAS, {'gamma': 0.5}, 'PAS', 'швидке спадання довіри (gamma = 0.5)'),
    'PAS-g1':     (NV.PAS, {'gamma': 1.0}, 'PAS', 'повна довіра до намірів (gamma = 1)'),
    'PAS-H7':     (NV.PAS, {'Hh': 7}, 'PAS', 'короткий горизонт (7 кроків)'),
    'PAS-H28':    (NV.PAS, {'Hh': 28}, 'PAS', 'довгий горизонт (28 кроків)'),
    'PAS-P16':    (NV.PAS, {'P': 16}, 'PAS', 'мала популяція фантомів (16)'),
    'PAS-P64':    (NV.PAS, {'P': 64}, 'PAS', 'велика популяція фантомів (64)'),
    'PAS-R1':     (NV.PAS, {'replan': 1}, 'PAS', 'перепланування на кожному кроці'),
    # BSS (гіпотези H3.2–H3.5)
    'BSS-prior':  (BSSFixed, {'mode': 'prior'}, 'BSS', 'лише модель апріору (без механізму сумніву)'),
    'BSS-unif':   (BSSFixed, {'mode': 'uniform'}, 'BSS', 'лише рівномірна модель (апріор ігнорується)'),
    'BSS-w04':    (NV.BSS, {'w1_0': 0.4}, 'BSS', 'апріорна довіра до запасної моделі 0.4'),
    'BSS-eta0':   (NV.BSS, {'eta': 0.0}, 'BSS', 'без глобального «виходу» до цілі (лише локальний крок)'),
    # PPS (гіпотеза H4.1): стійкість негативного результату до beta_s
    'PPS-b01':    (NV.PPS, {'beta_s': 0.1}, 'PPS', 'слабкі стоки (beta_s = 0.1)'),
    'PPS-b025':   (NV.PPS, {'beta_s': 0.25}, 'PPS', 'beta_s = 0.25'),
    'PPS-b1':     (NV.PPS, {'beta_s': 1.0}, 'PPS', 'beta_s = 1.0'),
    'PPS-b2':     (NV.PPS, {'beta_s': 2.0}, 'PPS', 'сильні стоки (beta_s = 2.0)'),
}
VARIANT_ORDER = list(VARIANTS)
BASE_NAMES = {c.name for c in (B.Lawnmower, B.Levy, B.PSO, B.GWO, B.ACO, B.Greedy, B.HEDAC, NV.SIFD, NV.PAS,
                               NV.BSS, NV.PPS, HB.GWO_PF, HB.ACO_IFD, HB.PSO_LS)}
# суфікси -C / -U зарезервовані для децентралізованих / необізнаних варіантів
assert not any(n.endswith(('-C', '-U')) for n in VARIANTS), 'назва варіанта не може закінчуватись на -C / -U'


def base_of(name):
    """Базовий метод для назви варіанта / децентралізованого варіанта."""
    n = name
    if n not in VARIANTS and n not in BASE_NAMES:
        for suf in ('-C', '-U'):
            if n.endswith(suf):
                n = n[:-len(suf)]
    if n in VARIANTS:
        return VARIANTS[n][2]
    return n
