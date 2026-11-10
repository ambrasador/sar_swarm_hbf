# -*- coding: utf-8 -*-
"""
Середовище пошуково-рятувальної операції (ПРО) для рою БПЛА.
Дискретна сітка W x H, байєсівська карта ймовірностей (occupancy grid),
неідеальний сенсор з POD, що залежить від рельєфу та відстані,
крива виживання постраждалих S(t) = exp(-(t/tau)^kappa).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, label, distance_transform_edt, uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

W = H = 50
N = W * H
MOVES = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
FP_R2 = 2                                               # слід сенсора: d^2 <= 2 (3x3)
FP_OFF = np.array([(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr * dr + dc * dc <= FP_R2])
FP_W = 1.0 - 0.4 * (FP_OFF ** 2).sum(1) / FP_R2
LANE = 3                                                # крок галсів для човникового сканування          # спадання POD від центру сліду
P0 = 0.70                                               # POD у центрі сліду на відкритій місцевості
TERRAIN_FACTOR = np.array([1.0, 0.7, 0.4])              # відкрита / чагарник / ліс
WIN_R = 5                                               # радіус сприйняття карти для PSO/GWO
TAU_S, KAPPA_S = 180.0, 1.5                             # параметри кривої виживання
T_MAX = 300
N_AGENTS = 5
K_VICTIMS = 3
ROW_EPS = 0.05                                          # маса «решти світу» в апріорі

RR, CC = np.divmod(np.arange(N), W)
RC = np.stack([RR, CC], 1).astype(float)


def survival(t):
    return np.exp(-(np.asarray(t, float) / TAU_S) ** KAPPA_S)


def idx(r, c):
    return r * W + c


class MapInstance:
    """Карта: рельєф, заборонені зони, апріорний розподіл, база, передобчислення."""

    def __init__(self, scenario, seed, lam_len=5.0):
        self.scenario = scenario
        self.seed = seed
        rng = np.random.default_rng(10_000 + seed * 7 + {'S1': 0, 'S2': 1, 'S3': 2}[scenario] * 1_000_003)
        self._gen_terrain(rng)
        self._gen_obstacles(rng)
        self._gen_prior(rng)
        self._precompute(lam_len)

    # ---------- генерація ----------
    def _gen_terrain(self, rng):
        noise = gaussian_filter(rng.standard_normal((H, W)), 4)
        forest_share = {'S1': 0.10, 'S2': 0.30, 'S3': 0.35}[self.scenario]
        shrub_share = {'S1': 0.15, 'S2': 0.20, 'S3': 0.20}[self.scenario]
        q_f = np.quantile(noise, 1 - forest_share)
        q_s = np.quantile(noise, 1 - forest_share - shrub_share)
        terr = np.zeros((H, W), int)
        terr[noise > q_s] = 1
        terr[noise > q_f] = 2
        self.terrain = terr.ravel()

    def _gen_obstacles(self, rng):
        free = np.ones((H, W), bool)
        n_zones = {'S1': 1, 'S2': 3, 'S3': 6}[self.scenario]
        for _ in range(n_zones):  # заборонені для польотів зони (ЛЕП, зони обстрілу тощо)
            if rng.random() < 0.5:
                r0, c0 = rng.integers(3, H - 12), rng.integers(3, W - 10)
                h, w = rng.integers(3, 9), rng.integers(3, 12)
                free[r0:r0 + h, c0:c0 + w] = False
            else:
                r0, c0, rad = rng.integers(5, H - 8), rng.integers(5, W - 5), rng.integers(2, 5)
                free[(RR.reshape(H, W) - r0) ** 2 + (CC.reshape(H, W) - c0) ** 2 <= rad * rad] = False
        base_r, base_c = H - 2, int(rng.integers(W // 4, 3 * W // 4))
        free[base_r - 1:base_r + 2, base_c - 1:base_c + 2] = True
        lab, _ = label(free, structure=np.ones((3, 3)))
        free = lab == lab[base_r, base_c]          # лишаємо тільки зв'язну компоненту бази
        self.free = free.ravel()
        self.base = idx(base_r, base_c)

    def _gen_prior(self, rng):
        g = np.zeros((H, W))
        R2, C2 = RR.reshape(H, W), CC.reshape(H, W)

        def gauss(r, c, s, w):
            return w * np.exp(-((R2 - r) ** 2 + (C2 - c) ** 2) / (2 * s * s))

        if self.scenario == 'S1':          # одна точка останнього спостереження (LKP)
            r, c = rng.integers(8, 28), rng.integers(10, 40)
            g += gauss(r, c, 6.0, 1.0)
        else:                              # мультимодальний (Lost Person Behavior)
            r, c = rng.integers(8, 30), rng.integers(8, 42)
            g += gauss(r, c, 4.0, 1.0) / (2 * np.pi * 16)
            # стежка: відрізок
            r1, c1, r2, c2 = rng.integers(3, 35), rng.integers(0, 10), rng.integers(3, 35), rng.integers(40, 50)
            tt = np.linspace(0, 1, 60)
            line = np.zeros((H, W))
            for a in tt:
                line += gauss(r1 + a * (r2 - r1), c1 + a * (c2 - c1), 1.8, 1.0)
            g += 1.3 * line / line.sum()
            # річка: синусоїдальна смуга
            rr0, amp, ph = rng.integers(10, 35), rng.uniform(2, 6), rng.uniform(0, 6.28)
            river_r = rr0 + amp * np.sin(np.linspace(0, 2 * np.pi, W) + ph)
            band = np.exp(-(R2 - river_r[None, :]) ** 2 / (2 * 1.5 ** 2))
            g += 1.0 * band / band.sum()
            # споруда
            r, c = rng.integers(5, 38), rng.integers(5, 45)
            s = gauss(r, c, 2.2, 1.0)
            g += 0.8 * s / s.sum()
        g = g.ravel() * self.free
        self.prior = g / g.sum()

    # ---------- передобчислення ----------
    def _precompute(self, lam_len):
        # сусіди (8-зв'язність)
        nbr = np.full((N, 8), -1, int)
        for k, (dr, dc) in enumerate(MOVES):
            r2, c2 = RR + dr, CC + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            j = np.where(ok, r2 * W + c2, 0)
            ok &= self.free[j] & self.free
            nbr[ok, k] = j[ok]
        self.nbr = nbr
        # слід сенсора
        F = len(FP_OFF)
        fp_idx = np.full((N, F), N, int)
        fp_p = np.zeros((N, F))
        for k, (dr, dc) in enumerate(FP_OFF):
            r2, c2 = RR + dr, CC + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            j = np.where(ok, r2 * W + c2, 0)
            ok &= self.free[j]
            fp_idx[ok, k] = j[ok]
            fp_p[ok, k] = P0 * TERRAIN_FACTOR[self.terrain[j[ok]]] * FP_W[k]
        self.fp_idx, self.fp_p = fp_idx, fp_p
        self.pdet = P0 * TERRAIN_FACTOR[self.terrain] * self.free        # виявлюваність клітинки
        # вікно сприйняття (Чебишев)
        offs = np.array([(dr, dc) for dr in range(-WIN_R, WIN_R + 1) for dc in range(-WIN_R, WIN_R + 1)])
        win = np.full((N, len(offs)), N, int)
        for k, (dr, dc) in enumerate(offs):
            r2, c2 = RR + dr, CC + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            j = np.where(ok, r2 * W + c2, 0)
            ok &= self.free[j]
            win[ok, k] = j[ok]
        self.win = win
        # найкоротші відстані (BFS, 8-зв'язність) між усіма парами вільних клітинок
        rows, cols = np.nonzero(nbr >= 0)
        A = coo_matrix((np.ones(len(rows)), (rows, nbr[rows, cols])), shape=(N, N)).tocsr()
        D = shortest_path(A, method='D', unweighted=True, directed=False)
        D[~np.isfinite(D)] = 9999
        self.apsp = D.astype(np.int16)
        # найближча вільна клітинка для довільної точки
        _, inds = distance_transform_edt(~self.free.reshape(H, W), return_indices=True)
        self.nearest_free = (inds[0] * W + inds[1]).ravel()
        # функція Гріна екранованого рівняння Пуассона (lambda*I + Dg - A4) G = I
        lam = 1.0 / lam_len ** 2
        r4, c4, v4 = [], [], []
        deg = np.zeros(N)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = RR + dr, CC + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            j = np.where(ok, r2 * W + c2, 0)
            ok &= self.free[j] & self.free
            r4.append(np.nonzero(ok)[0]); c4.append(j[ok])
            deg[ok] += 1
        r4, c4 = np.concatenate(r4), np.concatenate(c4)
        Mx = np.diag(lam + deg)
        Mx[r4, c4] -= 1.0
        self.G = np.linalg.inv(Mx).astype(np.float32)
        self.G[:, ~self.free] = 0
        self.G[~self.free, :] = 0

    def sample_victims(self, rng, k=K_VICTIMS):
        if self.scenario == 'S3':          # неточний апріор: 40% жертв поза моделлю
            out = []
            fr = np.nonzero(self.free)[0]
            for _ in range(k):
                if rng.random() < 0.6:
                    out.append(rng.choice(N, p=self.prior))
                else:
                    out.append(rng.choice(fr))
            return np.array(out)
        return rng.choice(N, size=k, replace=False, p=self.prior)

    def q0(self, k=K_VICTIMS, row=ROW_EPS):
        # «решта світу» (ROW): частка row маси рівномірно по області — стандартна практика
        # теорії пошуку; у S3 модель не знає, що реальна частка поза апріором становить 40%
        u = self.free / self.free.sum()
        return np.minimum(0.95, k * ((1 - row) * self.prior + row * u))


class SAREnv:
    def __init__(self, mp, victims, seed, M=N_AGENTS, T=T_MAX):
        self.mp, self.M, self.T = mp, M, T
        self.q = mp.q0(len(victims))
        self.q_total0 = self.q.sum()
        self.victims = np.array(victims)
        self.found_t = np.full(len(victims), -1)
        self.pos = np.full(M, mp.base)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.cpos = [0.0]
        self.glimpses = np.zeros(N + 1, int)
        self.path_len = 0

    # очікуване виявлення при огляді з кожної клітинки (для всієї карти)
    def gain_map(self, q=None):
        q = self.q if q is None else q
        qq = np.append(q, 0.0)
        g = (qq[self.mp.fp_idx] * self.mp.fp_p).sum(1)
        g[~self.mp.free] = 0
        return g

    def gain_at(self, cells, q=None):
        q = self.q if q is None else q
        qq = np.append(q, 0.0)
        return (qq[self.mp.fp_idx[cells]] * self.mp.fp_p[cells]).sum(-1)

    def step(self, new_pos):
        mp = self.mp
        new_pos = np.asarray(new_pos, int).copy()
        for i in range(self.M):
            if new_pos[i] != self.pos[i] and new_pos[i] not in mp.nbr[self.pos[i]]:
                new_pos[i] = self.pos[i]                     # некоректний хід — лишається на місці
        self.path_len += int((new_pos != self.pos).sum())
        self.pos = new_pos
        self.t += 1
        found_cells = []
        for i in range(self.M):
            fi, fp = mp.fp_idx[self.pos[i]], mp.fp_p[self.pos[i]]
            for v in np.nonzero(self.found_t < 0)[0]:
                hit = np.nonzero(fi == self.victims[v])[0]
                if len(hit) and self.rng.random() < fp[hit[0]]:
                    self.found_t[v] = self.t
                    found_cells.append(self.victims[v])
            ok = fi < N
            qi = self.q[fi[ok]]
            p = fp[ok]
            self.q[fi[ok]] = qi * (1 - p) / (1 - qi * p)     # байєсівське оновлення (негативна інформація)
            self.glimpses[fi[ok]] += 1
        for c in found_cells:
            self.q[c] = 0.0
        self.cpos.append(1.0 - self.q.sum() / self.q_total0)

    def done(self):
        return self.t >= self.T   # рій не знає кількості постраждалих — пошук до кінця бюджету

    def metrics(self):
        ft = self.found_t
        found = ft >= 0
        tt = np.where(found, ft, self.T)
        sws = np.where(found, survival(np.where(found, ft, 0)), 0.0).mean()
        uniq = (self.glimpses[:N] > 0).sum()
        cp = np.array(self.cpos + [self.cpos[-1]] * (self.T + 1 - len(self.cpos)))
        return dict(found_frac=found.mean(), t_first=tt.min(), t_all=tt.max(), t_mean=tt.mean(),
                    sws=sws, coverage=uniq / self.mp.free.sum(),
                    overlap=self.glimpses[:N].sum() / max(1, uniq), cpos_T=cp[-1],
                    cpos_100=cp[min(100, self.T)]), cp


# ---------------- допоміжні функції руху ----------------
def valid_nbrs(mp, c):
    nb = mp.nbr[c]
    return nb[nb >= 0]


def step_toward(mp, c, target, rng=None, tie=None):
    """Один крок найкоротшим шляхом (BFS) до цілі; tie — оцінки для розв'язання рівностей."""
    nb = valid_nbrs(mp, c)
    if len(nb) == 0:
        return c
    d = mp.apsp[target, nb].astype(float)
    if tie is not None:
        d = d - 1e-6 * (tie[nb] - tie[nb].min()) / (np.ptp(tie[nb]) + 1e-18)
    elif rng is not None:
        d = d + 1e-3 * rng.random(len(nb))
    return nb[np.argmin(d)]


def snap(mp, point):
    r = int(np.clip(round(point[0]), 0, H - 1))
    c = int(np.clip(round(point[1]), 0, W - 1))
    return mp.nearest_free[r * W + c]
