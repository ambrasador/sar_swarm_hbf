# -*- coding: utf-8 -*-
"""
Розширення середовища для експериментів v2 (env.py НЕ змінюється — `main` лишається відтворюваним).

  * apply_env_params      — параметри кривої виживання tau, kappa на час прогону (Е2-повтор);
  * sample_victims_flags  — та сама вибірка постраждалих, що й MapInstance.sample_victims,
                            плюс ознака «в межах моделі / поза моделлю» і керована частка rho_out (Е9);
  * SAREnvExt             — SAREnv з параметром маси «решти світу» row (модель знає / не знає rho_out);
  * SAREnvDyn             — рухомі постраждалі + дифузія карти переконань моделлю руху (Е3);
  * SAREnvComm            — відмови агентів (Е4): випадкові (lambda на агента на крок) або разові
                            (n агентів на кроці t_fail); фіксує, хто саме виявив постраждалого.

Усі класи за типових параметрів побітово відтворюють SAREnv (перевіряє tests/test_extensions.py).
"""
import zlib
import numpy as np
from scipy.sparse import csr_matrix

import env as E
from env import SAREnv, N, K_VICTIMS

DEFAULT_TAU_S, DEFAULT_KAPPA_S = float(E.TAU_S), float(E.KAPPA_S)


def apply_env_params(tau_s=None, kappa_s=None):
    """Встановлює параметри S(t) у модулі env (функція survival читає їх під час виклику)."""
    E.TAU_S = DEFAULT_TAU_S if tau_s is None else float(tau_s)
    E.KAPPA_S = DEFAULT_KAPPA_S if kappa_s is None else float(kappa_s)


def reset_env_params():
    apply_env_params(None, None)


def sample_victims_flags(mp, rng, k=K_VICTIMS, p_in=None):
    """
    Повторює MapInstance.sample_victims (ті самі виклики генератора) і повертає ознаки
    «постраждалий згенерований з апріору» (1) чи «поза моделлю» (0).
    p_in=None — поведінка оригіналу (S3: 0.6, S1/S2: усі з апріору без повторень);
    p_in задано — схема S3 з частками p_in / (1 - p_in) для будь-якої карти.
    """
    if mp.scenario == 'S3' or p_in is not None:
        pin = 0.6 if p_in is None else float(p_in)
        out, flags = [], []
        fr = np.nonzero(mp.free)[0]
        for _ in range(k):
            if rng.random() < pin:
                out.append(rng.choice(N, p=mp.prior)); flags.append(1)
            else:
                out.append(rng.choice(fr)); flags.append(0)
        return np.array(out), np.array(flags, int)
    v = rng.choice(N, size=k, replace=False, p=mp.prior)
    return v, np.ones(k, int)


class SAREnvExt(SAREnv):
    """SAREnv із керованою масою «решти світу» row у початковій карті (None — як у SAREnv)."""

    def __init__(self, mp, victims, seed, M=E.N_AGENTS, T=E.T_MAX, row=None, key=''):
        super().__init__(mp, victims, seed, M=M, T=T)
        self.key = key
        if row is not None:
            self.q = mp.q0(len(victims), row=float(row))
            self.q_total0 = self.q.sum()

    def extra_metrics(self):
        return {}


class SAREnvDyn(SAREnvExt):
    """
    Рухомі постраждалі. На початку кожного кроку незнайдений постраждалий з імовірністю p_move
    переходить у сусідню вільну клітинку з імовірністю, пропорційною  prior(n) + attract_eps * mean(prior)
    (тяжіння до стежок/річки/споруд апріору). Траєкторії постраждалих екзогенні: генеруються
    заздалегідь окремим генератором за ключем екземпляра — однакові для всіх методів (парний дизайн).
    Карта переконань перед оглядом проходить крок прогнозу тією самою (правильною) моделлю руху:
        q <- (1 - p) q + p K^T q,  з обмеженням [0, 0.95].
    p_move = 0 — побітово SAREnv.
    """

    def __init__(self, mp, victims, seed, M=E.N_AGENTS, T=E.T_MAX, row=None, key='',
                 p_move=0.0, attract_eps=1.0):
        super().__init__(mp, victims, seed, M=M, T=T, row=row, key=key)
        self.p_move = float(p_move)
        self.victim_traj = np.repeat(self.victims[None, :], T + 1, 0)
        if self.p_move > 0:
            self.KT, self._nb_w = self._transition(mp, float(attract_eps))
            vrng = np.random.default_rng(zlib.crc32((key + '|dyn').encode()))
            cur = self.victims.copy()
            for t in range(1, T + 1):
                for v in range(len(cur)):
                    u = vrng.random()
                    nb, w = self._nb_w[int(cur[v])]
                    j = vrng.choice(len(nb), p=w)
                    if u < self.p_move:
                        cur[v] = nb[j]
                self.victim_traj[t] = cur

    @staticmethod
    def _transition(mp, attract_eps):
        """Матриця переходів моделі руху (кешується на об'єкті карти)."""
        cache = mp.__dict__.setdefault('_dyn_cache', {})
        if attract_eps in cache:
            return cache[attract_eps]
        rows, cols, vals = [], [], []
        pr = mp.prior
        mean_pr = pr[mp.free].mean()
        for c in np.nonzero(mp.free)[0]:
            nb = mp.nbr[c]
            nb = nb[nb >= 0]
            if len(nb) == 0:
                rows.append(c); cols.append(c); vals.append(1.0)
                continue
            w = pr[nb] + attract_eps * mean_pr
            w = w / w.sum()
            rows += [c] * len(nb); cols += nb.tolist(); vals += w.tolist()
        K = csr_matrix((vals, (rows, cols)), shape=(N, N))
        nb_w = {}
        for c in np.nonzero(mp.free)[0]:
            s, e = K.indptr[c], K.indptr[c + 1]
            nb_w[int(c)] = (K.indices[s:e].copy(), K.data[s:e].copy())
        cache[attract_eps] = (K.T.tocsr(), nb_w)
        return cache[attract_eps]

    def step(self, new_pos):
        if self.p_move > 0:
            t_next = self.t + 1
            alive_v = self.found_t < 0
            self.victims = np.where(alive_v, self.victim_traj[t_next], self.victims)
            self.q = np.clip((1.0 - self.p_move) * self.q + self.p_move * (self.KT @ self.q), 0.0, 0.95)
        super().step(new_pos)

    def extra_metrics(self):
        moved = int((self.victim_traj[1:] != self.victim_traj[:-1]).any(1).sum()) if self.p_move > 0 else 0
        return dict(victim_moves=moved)


class SAREnvComm(SAREnvExt):
    """
    Середовище з відмовами агентів. Агент i «мертвий» на кроках s >= death_t[i]: не рухається,
    не оглядає, не бере участі у зв'язку. Моменти відмов генеруються заздалегідь окремим
    генератором за ключем екземпляра — однакові для всіх методів (парний дизайн).
      fail = None                                    — відмов немає (побітово SAREnv);
      fail = {"type": "rand", "lambda": 0.002}       — незалежна відмова КОЖНОГО агента з імовірністю
                                                       lambda на кожному кроці (очікувана кількість
                                                       відмов M(1-(1-lambda)^T));
      fail = {"type": "fixed", "t": 100, "n": 2}     — n випадково обраних агентів відмовляють на кроці t.
    Додатково фіксується finder[v] — агент, що виявив постраждалого v (для децентралізованих знань).
    """

    def __init__(self, mp, victims, seed, M=E.N_AGENTS, T=E.T_MAX, row=None, key='', fail=None):
        super().__init__(mp, victims, seed, M=M, T=T, row=row, key=key)
        self.death_t = np.full(M, T + 10, int)
        if fail:
            frng = np.random.default_rng(zlib.crc32((key + '|fail').encode()))
            if fail['type'] == 'rand':
                lam = float(fail['lambda'])
                u = frng.random((M, T))
                for i in range(M):
                    hit = np.nonzero(u[i] < lam)[0]
                    if len(hit):
                        self.death_t[i] = int(hit[0]) + 1
            elif fail['type'] == 'fixed':
                who = frng.choice(M, size=int(fail['n']), replace=False)
                self.death_t[who] = int(fail['t'])
            else:
                raise ValueError(f'невідомий тип відмов: {fail}')
        self.alive = np.ones(M, bool)
        self.finder = np.full(len(victims), -1, int)

    def step(self, new_pos):
        mp = self.mp
        t_next = self.t + 1
        self.alive = self.death_t > t_next
        new_pos = np.asarray(new_pos, int).copy()
        for i in range(self.M):
            if not self.alive[i]:
                new_pos[i] = self.pos[i]
                continue
            if new_pos[i] != self.pos[i] and new_pos[i] not in mp.nbr[self.pos[i]]:
                new_pos[i] = self.pos[i]                     # некоректний хід — лишається на місці
        self.path_len += int((new_pos != self.pos).sum())
        self.pos = new_pos
        self.t += 1
        found_cells = []
        for i in range(self.M):
            if not self.alive[i]:
                continue
            fi, fp = mp.fp_idx[self.pos[i]], mp.fp_p[self.pos[i]]
            for v in np.nonzero(self.found_t < 0)[0]:
                hit = np.nonzero(fi == self.victims[v])[0]
                if len(hit) and self.rng.random() < fp[hit[0]]:
                    self.found_t[v] = self.t
                    self.finder[v] = i
                    found_cells.append(self.victims[v])
            ok = fi < N
            qi = self.q[fi[ok]]
            p = fp[ok]
            self.q[fi[ok]] = qi * (1 - p) / (1 - qi * p)     # байєсівське оновлення (як у SAREnv)
            self.glimpses[fi[ok]] += 1
        for c in found_cells:
            self.q[c] = 0.0
        self.cpos.append(1.0 - self.q.sum() / self.q_total0)

    def extra_metrics(self):
        dead = self.death_t <= self.T
        return dict(n_failed=int(dead.sum()),
                    fail_times=[int(x) for x in np.where(dead, self.death_t, -1)])
