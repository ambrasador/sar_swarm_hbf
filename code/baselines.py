# -*- coding: utf-8 -*-
"""Існуючі (базові) методи для порівняння: LAWN, LEVY, PSO, GWO, ACO, GREEDY."""
import numpy as np
from env import (N, W, H, RC, LANE, valid_nbrs, step_toward, snap)


class Method:
    name = 'BASE'
    def reset(self, env, rng):
        self.rng = rng
    def act(self, env):
        raise NotImplementedError


class Lawnmower(Method):
    """Класичне човникове (бустрофедон) сканування смуг, закріплених за агентами."""
    name = 'LAWN'
    def reset(self, env, rng):
        super().reset(env, rng)
        mp, M = env.mp, env.M
        strips = np.linspace(0, W, M + 1).astype(int)
        self.wps, self.k = [], np.zeros(M, int)
        for i in range(M):
            a, b = strips[i], strips[i + 1]
            lanes = list(range(a + 1, b, LANE))
            if lanes[-1] < b - 2:
                lanes.append(b - 2)
            pts = []
            for j, lc in enumerate(lanes):
                rows = (H - 1, 0) if j % 2 == 0 else (0, H - 1)
                pts += [snap(mp, (rows[0], lc)), snap(mp, (rows[1], lc))]
            self.wps.append(pts)
    def act(self, env):
        out = env.pos.copy()
        for i in range(env.M):
            if env.pos[i] == self.wps[i][self.k[i]]:
                self.k[i] = (self.k[i] + 1) % len(self.wps[i])
            out[i] = step_toward(env.mp, env.pos[i], self.wps[i][self.k[i]], self.rng)
        return out


class Levy(Method):
    """Випадкові блукання Леві (показник mu=2)."""
    name = 'LEVY'
    def reset(self, env, rng):
        super().reset(env, rng)
        self.dir = rng.integers(8, size=env.M)
        self.left = np.zeros(env.M, int)
    def act(self, env):
        mp, out = env.mp, env.pos.copy()
        for i in range(env.M):
            if self.left[i] <= 0 or mp.nbr[env.pos[i], self.dir[i]] < 0:
                ok = np.nonzero(mp.nbr[env.pos[i]] >= 0)[0]
                self.dir[i] = self.rng.choice(ok)
                self.left[i] = int(min(25, np.floor(self.rng.pareto(1.0) + 1)))
            out[i] = mp.nbr[env.pos[i], self.dir[i]]
            self.left[i] -= 1
        return out


def top_distinct(cells, vals, k, min_sep=3):
    order = np.argsort(-vals)
    chosen = []
    for o in order:
        c = cells[o]
        if all(np.abs(RC[c] - RC[x]).max() >= min_sep for x in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    while len(chosen) < k:
        chosen.append(chosen[-1])
    return chosen


class PSO(Method):
    """Робототехнічний (динамічний) PSO: pbest/gbest переоцінюються на поточній карті ймовірностей."""
    name = 'PSO'
    w, c1, c2, vmax = 0.72, 1.49, 1.49, 3.0
    def reset(self, env, rng):
        super().reset(env, rng)
        self.x = np.repeat(RC[env.mp.base][None], env.M, 0)
        self.v = rng.uniform(-1, 1, (env.M, 2))
        self.pbest = np.full(env.M, env.mp.base)
    def update_bests(self, env, F):
        Fd = np.append(F, -np.inf)
        for i in range(env.M):
            win = env.mp.win[env.pos[i]]
            j = win[np.argmax(Fd[win])]
            if Fd[j] > Fd[self.pbest[i]]:
                self.pbest[i] = j
        self.gbest = self.pbest[np.argmax(Fd[self.pbest])]
    def pso_target(self, i, env):
        r1, r2 = self.rng.random(2), self.rng.random(2)
        self.v[i] = (self.w * self.v[i] + self.c1 * r1 * (RC[self.pbest[i]] - self.x[i])
                     + self.c2 * r2 * (RC[self.gbest] - self.x[i]))
        n = np.linalg.norm(self.v[i])
        if n > self.vmax:
            self.v[i] *= self.vmax / n
        return snap(env.mp, self.x[i] + self.v[i])
    def act(self, env):
        F = env.gain_map()
        self.update_bests(env, F)
        out = env.pos.copy()
        for i in range(env.M):
            tgt = self.pso_target(i, env)
            out[i] = step_toward(env.mp, env.pos[i], tgt, self.rng) if tgt != env.pos[i] else \
                self.rng.choice(valid_nbrs(env.mp, env.pos[i]))
            self.x[i] = RC[out[i]]
        return out


class GWO(Method):
    """Сірий вовк: лідери alpha/beta/delta — найкращі відомі позиції рою; a = 2(1 - t/T)."""
    name = 'GWO'
    def reset(self, env, rng):
        super().reset(env, rng)
        self.known = set([env.mp.base])
    def act(self, env):
        mp, F = env.mp, env.gain_map()
        for i in range(env.M):
            w = mp.win[env.pos[i]]
            self.known.update(w[w < N].tolist())
        cells = np.fromiter(self.known, int)
        leaders = top_distinct(cells, F[cells], 3)
        a = 2.0 * (1 - env.t / env.T)
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


class ACO(Method):
    """Мурашиний алгоритм з відштовхувальним феромоном (покриття + евристика ймовірності)."""
    name = 'ACO'
    alpha, beta, rho = 1.5, 2.0, 0.02
    def reset(self, env, rng):
        super().reset(env, rng)
        self.tau = np.zeros(N + 1)
    def act(self, env):
        mp, F = env.mp, env.gain_map()
        out = env.pos.copy()
        for i in range(env.M):
            nb = valid_nbrs(mp, env.pos[i])
            att = (F[nb] + 1e-12) ** self.beta * (1 + self.tau[nb]) ** (-self.alpha)
            out[i] = self.rng.choice(nb, p=att / att.sum())
            self.tau[mp.fp_idx[out[i]]] += mp.fp_p[out[i]] / 0.7   # депонування у сліді
        self.tau *= (1 - self.rho)
        self.tau[N] = 0
        return out


class Greedy(Method):
    """Інформаційно-жадібний координований метод (послідовне віртуальне виснаження карти)."""
    name = 'GREEDY'
    def act(self, env):
        mp = env.mp
        qv = np.append(env.q, 0.0)
        out = env.pos.copy()
        for i in self.rng.permutation(env.M):
            nb = valid_nbrs(mp, env.pos[i])
            g = (qv[mp.fp_idx[nb]] * mp.fp_p[nb]).sum(1) + 1e-15 * self.rng.random(len(nb))
            c = nb[np.argmax(g)]
            out[i] = c
            fi, p = mp.fp_idx[c], mp.fp_p[c]
            qv[fi] = qv[fi] * (1 - p) / (1 - qv[fi] * p)
        return out


class HEDAC(Method):
    """
    Спрощена реалізація HEDAC (Heat Equation Driven Area Coverage, Ivić et al., 2017/2019)
    для порівняння: поле — розв'язок рівняння Гельмгольца з джерелом q*POD
    (непокрита виявлювана ймовірність), агенти рухаються за градієнтом поля; уникнення
    зіткнень — заборона займати клітинку/сусідство вже обраних позицій.
    Використовує ту саму функцію Гріна та те саме мікроправило (з інерцією), що й PPS,
    тож порівняння HEDAC* vs PPS є абляцією механізму «агент-стік».
    """
    name = 'HEDAC*'
    mu = 0.15

    def reset(self, env, rng):
        super().reset(env, rng)
        self.prev = np.zeros((env.M, 2))

    def act(self, env):
        mp = env.mp
        phi = mp.G @ (env.q * mp.pdet).astype(np.float32)
        out, chosen = env.pos.copy(), []
        for i in self.rng.permutation(env.M):
            c0 = env.pos[i]
            nb = valid_nbrs(mp, c0)
            vals = phi[nb].astype(float)
            d = RC[nb] - RC[c0]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            vals = vals + self.mu * (vals.std() + 1e-18) * (d @ self.prev[i]) + 1e-12 * self.rng.random(len(nb))
            if chosen:
                bad = np.zeros(len(nb), bool)
                for c in chosen:
                    bad |= np.abs(RC[nb] - RC[c]).max(1) <= 1
                if (~bad).any():
                    vals[bad] = -np.inf
            c = nb[np.argmax(vals)]
            self.prev[i] = (RC[c] - RC[c0]) / np.linalg.norm(RC[c] - RC[c0])
            out[i] = c
            chosen.append(c)
        return out
