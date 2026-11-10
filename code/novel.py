# -*- coding: utf-8 -*-
"""
Три нові моделі ройового пошуку для ПРО:
  PPS  — Пуассонівський потенціальний рій (Poisson Potential Swarm)
  SIFD — Рій ідеального вільного розподілу з кривою виживання (Survival-weighted Ideal Free Distribution)
  PAS  — Рій фантомних агентів зі згасаючою довірою до намірів (Phantom-Agent Swarm)
"""
import numpy as np
from scipy.ndimage import uniform_filter
from env import (N, W, H, RC, survival, valid_nbrs, step_toward)
from baselines import Method


# =====================================================================================
# 1. PPS — Poisson Potential Swarm
# =====================================================================================
class PPS(Method):
    """
    Поле привабливості phi — розв'язок екранованого рівняння Пуассона на графі вільних клітинок:
        (lambda*I - L) phi = s - sum_j sigma * delta_{x_j},     s = q * POD
    Джерело — виявлювана ймовірнісна маса, агенти — «стоки» рівної потужності sigma = beta_s*sum(s)/M.
    Кожен агент піднімається градієнтом поля phi_i, з якого вилучено власний стік.
    Властивість: у зонах без маси (s=0) phi не має додатних локальних максимумів (дискретний
    принцип максимуму), тож агент не застрягає в уже обстежених зонах; стоки дають неявне
    розведення агентів без явних правил відштовхування.
    """
    name = 'PPS'
    beta_s, mu = 0.5, 0.15

    def reset(self, env, rng):
        super().reset(env, rng)
        self.prev = np.zeros((env.M, 2))

    def fields(self, env):
        mp = env.mp
        s = (env.q * mp.pdet).astype(np.float32)
        phi_b = mp.G @ s
        sigma = self.beta_s * s.sum() / env.M
        return phi_b, sigma

    def act(self, env):
        mp = env.mp
        phi_b, sigma = self.fields(env)
        cur = env.pos.copy()
        sink = sigma * mp.G[cur].sum(0)
        out = cur.copy()
        for i in self.rng.permutation(env.M):
            phi_i = phi_b - (sink - sigma * mp.G[cur[i]])
            nb = valid_nbrs(mp, cur[i])
            vals = phi_i[nb].astype(float)
            d = RC[nb] - RC[cur[i]]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            vals = vals + self.mu * (vals.std() + 1e-18) * (d @ self.prev[i]) + 1e-12 * self.rng.random(len(nb))
            c = nb[np.argmax(vals)]
            self.prev[i] = RC[c] - RC[cur[i]]
            self.prev[i] /= np.linalg.norm(self.prev[i])
            sink += sigma * (mp.G[c] - mp.G[cur[i]])       # послідовне оновлення стоків
            cur[i] = c
            out[i] = c
        return out


# =====================================================================================
# 2. SIFD — Survival-weighted Ideal Free Distribution Swarm
# =====================================================================================
class SIFD(Method):
    """
    Макрорівень (екологія добування їжі): область розбито на ділянки k. Для агента i
        R_ik = V_k * S(t + d_ik) / ((n_k + 1)^m * (d_ik + h)),
    де V_k — виявлювана маса ділянки, d_ik — геодезична відстань, S — крива виживання,
    n_k — кількість інших агентів на ділянці (інтерференція Сазерленда), h — час «обробки».
    Розподіл агентів — ітеративна найкраща відповідь (гра-конгестія з індивідуальними
    виграшами, рівновага за Мілчтайхом) з гістерезисом. Мікрорівень: локальний
    жадібний пошук у межах ділянки з відштовхуванням від напарників.
    """
    name = 'SIFD'
    P, m, h, hyst, realloc = 10, 1.0, 6.0, 0.15, 4

    def reset(self, env, rng):
        super().reset(env, rng)
        mp = env.mp
        kp = -(-W // self.P)
        self.pid = (np.arange(N) // W // self.P) * kp + (np.arange(N) % W) // self.P
        self.K = kp * (-(-H // self.P))
        self.cells = [np.nonzero((self.pid == k) & mp.free)[0] for k in range(self.K)]
        self.assign = np.full(env.M, -1)

    def patch_info(self, env, F):
        V = np.bincount(self.pid, weights=env.q * env.mp.pdet, minlength=self.K)
        tgt = np.array([c[np.argmax(F[c])] if len(c) else env.mp.base for c in self.cells])
        return V, tgt

    def allocate(self, env, V, tgt):
        mp, M = env.mp, env.M
        d = mp.apsp[np.ix_(env.pos, tgt)].astype(float)
        base_u = V[None, :] * survival(env.t + d) / (d + self.h)
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

    def micro(self, env, i, k, F, chosen):
        """Мікроправило всередині ділянки: жадібний крок з відштовхуванням."""
        mp = env.mp
        nb = valid_nbrs(mp, env.pos[i])
        inside = nb[self.pid[nb] == k]
        if len(inside):
            nb = inside
        sc = F[nb].copy()
        for c in chosen:
            close = np.abs(RC[nb] - RC[c]).max(1) <= 2
            sc[close] *= 0.5
        return nb[np.argmax(sc + 1e-15 * self.rng.random(len(nb)))]

    def act(self, env):
        F = env.gain_map()
        V, tgt = self.patch_info(env, F)
        if env.t % self.realloc == 0 or (self.assign < 0).any():
            self.allocate(env, V, tgt)
        out, chosen = env.pos.copy(), []
        for i in self.rng.permutation(env.M):
            k = self.assign[i]
            if self.pid[env.pos[i]] != k:
                out[i] = step_toward(env.mp, env.pos[i], tgt[k], tie=F)
            else:
                out[i] = self.micro(env, i, k, F, chosen)
            chosen.append(out[i])
        return out


# =====================================================================================
# 3. PAS — Phantom-Agent Swarm with decaying intent trust
# =====================================================================================
class PAS(Method):
    """
    Кожен реальний агент породжує популяцію P «фантомів» — уявних траєкторій довжини Hh,
    що еволюціонують за схемою крос-ентропійного відбору (2 покоління). Фантоми оцінюються
    на віртуальній карті, в якій уже «виконано» оголошені наміри інших агентів з довірою
    gamma^tau (що далі в майбутнє — то менша довіра). Критерій — очікувана кількість
    виявлень, зважена кривою виживання, плюс термінальна оцінка перспективності.
    Агент виконує перші кроки найкращого фантома та транслює план як намір (receding horizon).
    """
    name = 'PAS'
    P, Hh, replan, gamma, temp, omega = 32, 14, 3, 0.9, 0.6, 0.5

    def reset(self, env, rng):
        super().reset(env, rng)
        self.plans = [[] for _ in range(env.M)]
        self.round = 0

    def virtual_q(self, env, i):
        mp = env.mp
        qv = np.append(env.q, 0.0)
        for j in range(env.M):
            if j == i or not self.plans[j]:
                continue
            cells = np.array(self.plans[j][:self.Hh])
            fi = mp.fp_idx[cells]
            pp = mp.fp_p[cells] * (self.gamma ** np.arange(1, len(cells) + 1))[:, None]
            for l in range(len(cells)):
                v = qv[fi[l]]
                qv[fi[l]] = v * (1 - pp[l]) / (1 - v * pp[l])
        qv[N] = 0
        return qv

    def sample(self, mp, starts, g, prefix=None):
        P = len(starts)
        L = self.Hh
        k0 = 0 if prefix is None else prefix.shape[1]
        paths = np.zeros((P, L), int)
        if prefix is not None:
            paths[:, :k0] = prefix
        visited = np.zeros((P, L + 1), int)
        visited[:, 0] = starts
        visited[:, 1:k0 + 1] = paths[:, :k0]
        pos = paths[:, k0 - 1] if k0 > 0 else starts.copy()
        gd = np.append(g, 0.0)
        ar = np.arange(P)
        for t in range(k0, L):
            nb = mp.nbr[pos]
            valid = nb >= 0
            nbv = np.where(valid, nb, N)
            gv = gd[nbv]
            rev = (nbv[:, :, None] == visited[:, None, :t + 1]).any(-1)
            gv = np.where(rev, gv * 0.1, gv)
            mu = gv.max(1, keepdims=True)
            sd = gv.std(1, keepdims=True) + 1e-15
            z = np.where(valid, (gv - mu) / sd / self.temp, -np.inf)
            pr = np.exp(z)
            pr /= pr.sum(1, keepdims=True)
            u = self.rng.random((P, 1))
            ch = (pr.cumsum(1) < u).sum(1).clip(0, 7)
            ch = np.where(valid[ar, ch], ch, np.argmax(pr, 1))
            pos = nb[ar, ch]
            paths[:, t] = pos
            visited[:, t + 1] = pos
        return paths

    def evaluate(self, mp, paths, qv, t0, blur):
        P = len(paths)
        qph = np.tile(qv, (P, 1))
        rows = np.arange(P)[:, None]
        tot = np.zeros(P)
        for t in range(self.Hh):
            c = paths[:, t]
            fi, pp = mp.fp_idx[c], mp.fp_p[c]
            v = qph[rows, fi]
            tot += survival(t0 + t + 1) * (v * pp).sum(1)
            qph[rows, fi] = v * (1 - pp) / (1 - v * pp)
        tot += self.omega * self.Hh * survival(t0 + self.Hh) * blur[paths[:, -1]]
        return tot

    def plan_agent(self, env, i):
        mp = env.mp
        qv = self.virtual_q(env, i)
        g = env.gain_map(qv[:N])
        blur = uniform_filter(g.reshape(H, W), size=11, mode='constant').ravel()
        start = env.pos[i]
        paths = self.sample(mp, np.full(self.P, start), g)
        if self.plans[i] and len(self.plans[i]) >= 2:        # «теплий старт» — продовження попереднього плану
            prev = np.array(self.plans[i][:self.Hh])
            warm = self.sample(mp, np.array([start]), g, prefix=prev[None, :min(len(prev), self.Hh - 1)])
            paths = np.vstack([paths, warm])
        sc = self.evaluate(mp, paths, qv, env.t, blur)
        n_el = max(2, len(paths) // 4)
        elite = paths[np.argsort(-sc)[:n_el]]
        kids = []
        for k in (3, 7, 10):                                 # друге покоління: мутація суфіксів еліти
            par = elite[self.rng.integers(len(elite), size=self.P // 3)]
            kids.append(self.sample(mp, np.full(len(par), start), g, prefix=par[:, :k]))
        allp = np.vstack([elite] + kids)
        sc2 = self.evaluate(mp, allp, qv, env.t, blur)
        self.plans[i] = allp[np.argmax(sc2)].tolist()

    def act(self, env):
        if env.t % self.replan == 0 or any(len(p) == 0 for p in self.plans):
            order = np.roll(np.arange(env.M), self.round)
            self.round += 1
            for i in order:
                self.plan_agent(env, i)
        out = env.pos.copy()
        for i in range(env.M):
            nxt = self.plans[i].pop(0)
            out[i] = nxt if (nxt in env.mp.nbr[env.pos[i]]) else step_toward(env.mp, env.pos[i], nxt, self.rng)
        return out


# =====================================================================================
# 4. BSS — Bayesian Skeptic Swarm (рій з байєсівською довірою до апріору)
# =====================================================================================
class BSS(Method):
    """
    Рій підтримує дві конкуруючі моделі світу, обидві оновлюються кожним оглядом:
      M0 — «апріор надійний»: карта q0 (профіль зниклого, LKP тощо);
      M1 — «апріор ненадійний»: рівномірна карта q1.
    Апостеріорна довіра w_m оновлюється за правдоподібністю спостережень
    (виявлення в клітинці c: q_m(c)p; невиявлення: 1 - q_m(c)p).
    Поділ праці: n_s = round(M * w1) агентів-«скептиків» шукають за q1 (невкрита територія),
    решта — «вірян» — за модельно-усередненою картою  q_bar = w0 q0 + w1 q1.
    Мікроправило обох ролей: координований жадібний крок з глобальним «виходом» до цілі
    argmax G(c)/(d(x,c)+h), якщо локальний виграш < eta * G(c*).
    """
    name = 'BSS'
    w1_0, h, eta = 0.1, 4.0, 0.3

    def reset(self, env, rng):
        super().reset(env, rng)
        mp = env.mp
        u = mp.free / mp.free.sum()
        self.qm = [env.q.copy(), np.minimum(0.95, env.q_total0 * u)]
        self.logw = np.log([1 - self.w1_0, self.w1_0])
        self.skeptic = np.zeros(env.M, bool)
        self.seen_t = 0

    def observe(self, env):
        if env.t == self.seen_t:
            return
        mp = env.mp
        newly = set(env.victims[env.found_t == env.t].tolist())
        for i in range(env.M):
            fi, fp = mp.fp_idx[env.pos[i]], mp.fp_p[env.pos[i]]
            ok = fi < N
            fi, fp = fi[ok], fp[ok]
            det = np.array([c in newly for c in fi])
            for m in range(2):
                q = self.qm[m][fi]
                lik = np.where(det, q * fp, 1 - q * fp)
                self.logw[m] += np.log(np.maximum(lik, 1e-300)).sum()
                self.qm[m][fi] = np.where(det, 0.0, q * (1 - fp) / (1 - q * fp))
            for c in fi[det]:
                newly.discard(c)
        self.logw -= self.logw.max()
        self.seen_t = env.t

    def weights(self):
        w = np.exp(self.logw)
        return w / w.sum()

    def act(self, env):
        mp = env.mp
        self.observe(env)
        w = self.weights()
        qbar = w[0] * self.qm[0] + w[1] * self.qm[1]
        maps = {False: qbar, True: self.qm[1]}
        # --- розподіл ролей (з гістерезисом: змінюємо лише кількість) ---
        n_s = int(np.floor(env.M * w[1] + 0.5))
        Gb = env.gain_map(qbar)
        cur_s = self.skeptic.sum()
        if cur_s < n_s:          # у скептики переходять агенти з найменшим локальним виграшем
            cand = np.nonzero(~self.skeptic)[0]
            self.skeptic[cand[np.argsort(Gb[env.pos[cand]])[:n_s - cur_s]]] = True
        elif cur_s > n_s:
            cand = np.nonzero(self.skeptic)[0]
            self.skeptic[cand[np.argsort(-Gb[env.pos[cand]])[:cur_s - n_s]]] = False
        Gmap = {False: Gb, True: env.gain_map(self.qm[1])}
        dep = np.ones(N + 1)
        out = env.pos.copy()
        for i in self.rng.permutation(env.M):
            role = bool(self.skeptic[i])
            q = np.append(maps[role], 0.0) * dep
            c0 = env.pos[i]
            nb = valid_nbrs(mp, c0)
            g = (q[mp.fp_idx[nb]] * mp.fp_p[nb]).sum(1)
            Gm = Gmap[role]
            score = Gm / (mp.apsp[c0].astype(float) + self.h)
            score[~mp.free] = -1
            cstar = int(np.argmax(score))
            if g.max() < self.eta * Gm[cstar] and cstar != c0:
                c = step_toward(mp, c0, cstar, tie=Gm)
            else:
                c = nb[np.argmax(g + 1e-15 * self.rng.random(len(nb)))]
            out[i] = c
            fi, fp = mp.fp_idx[c], mp.fp_p[c]
            dep[fi] *= (1 - fp)
        return out
