# -*- coding: utf-8 -*-
"""
Три гібридні моделі:
  GWO-PF  — Сірий вовк з лідерами на максимумах пуассонівського поля (GWO + PPS)
  ACO-IFD — Мурашиний мікропошук у ділянках ідеального вільного розподілу (SIFD + ACO)
  PSO-LS  — PSO з режимами «Леві-стрибок / човникове сканування» за локальним контрастом (PSO + Lévy + LAWN)
"""
import numpy as np
from scipy.ndimage import uniform_filter
from env import (N, W, H, RC, valid_nbrs, step_toward, snap)
from baselines import Method, PSO
from novel import SIFD


class GWO_PF(Method):
    """
    Децентралізований GWO: для кожного вовка i лідери alpha, beta, delta — три
    найвищі локальні максимуми його власного поля phi_i (PPS, стоки інших агентів),
    знайдені з придушенням немаксимумів. Замість центроїда X1..X3 (який у мультимодальному
    просторі потрапляє в «порожнечу» між модами) обирається один лідер рулеткою з
    ієрархічними вагами 0.5/0.3/0.2. Параметр a адаптується не за номером ітерації,
    а за залишковою масою: a = 2 * (q_sum / q_sum0)^0.5.
    """
    name = 'GWO-PF'
    beta_s, nms_r, retarget = 0.5, 6, 3
    w_h = np.array([0.5, 0.3, 0.2])          # ієрархічні ваги alpha/beta/delta (рулетка замість центроїда)

    def reset(self, env, rng):
        super().reset(env, rng)
        self.tgt = np.full(env.M, -1)

    def leaders(self, phi, free):
        f = np.where(free, phi, -np.inf).copy()
        out = []
        for _ in range(3):
            c = int(np.argmax(f))
            out.append(c)
            f[np.abs(RC - RC[c]).max(1) <= self.nms_r] = -np.inf
        return out

    def act(self, env):
        mp = env.mp
        s = (env.q * mp.pdet).astype(np.float32)
        phi_b = mp.G @ s
        sigma = self.beta_s * s.sum() / env.M
        cur = env.pos.copy()
        sink = sigma * mp.G[cur].sum(0)
        a = 2.0 * np.sqrt(max(env.q.sum(), 1e-12) / env.q_total0)
        out = cur.copy()
        for i in self.rng.permutation(env.M):
            phi_i = phi_b - (sink - sigma * mp.G[cur[i]])
            if env.t % self.retarget == 0 or self.tgt[i] < 0 or self.tgt[i] == cur[i]:
                X = RC[cur[i]]
                L = self.leaders(phi_i, mp.free)[self.rng.choice(3, p=self.w_h)]
                A = 2 * a * self.rng.random(2) - a
                C = 2 * self.rng.random(2)
                self.tgt[i] = snap(mp, RC[L] - A * np.abs(C * RC[L] - X))
            if self.tgt[i] == cur[i]:
                nb = valid_nbrs(mp, cur[i])
                c = nb[np.argmax(phi_i[nb])]
            else:
                c = step_toward(mp, cur[i], self.tgt[i], tie=phi_i)
            sink += sigma * (mp.G[c] - mp.G[cur[i]])
            cur[i] = c
            out[i] = c
        return out


class ACO_IFD(SIFD):
    """
    Макрорівень — SIFD (ідеальний вільний розподіл з виживанням); мікрорівень — мурашиний
    вибір з байєсівськи узгодженим феромоном: депонування пропорційне фактичній POD огляду,
    тому в зонах з низькою виявлюваністю (ліс) феромону менше і повторні огляди заохочуються.
    """
    name = 'ACO-IFD'
    alpha, beta, rho = 1.5, 2.0, 0.03

    def reset(self, env, rng):
        super().reset(env, rng)
        self.tau = np.zeros(N + 1)

    def micro(self, env, i, k, F, chosen):
        mp = env.mp
        nb = valid_nbrs(mp, env.pos[i])
        inside = nb[self.pid[nb] == k]
        if len(inside):
            nb = inside
        att = (F[nb] + 1e-12) ** self.beta * (1 + self.tau[nb]) ** (-self.alpha)
        for c in chosen:
            att[np.abs(RC[nb] - RC[c]).max(1) <= 1] *= 0.3
        return self.rng.choice(nb, p=att / att.sum())

    def act(self, env):
        out = super().act(env)
        mp = env.mp
        for c in out:
            self.tau[mp.fp_idx[c]] += mp.fp_p[c] / 0.7
        self.tau *= (1 - self.rho)
        self.tau[N] = 0
        return out


class PSO_LS(PSO):
    """
    Перемикання режимів за локальною статистикою карти:
      * застій (зібране виявлення за останні k кроків < theta_s * середнього по карті) -> Леві-стрибок
        до найперспективнішої з 8 Леві-цілей (оцінка — згладжена карта);
      * низький контраст (CV < theta_cv у вікні) -> локальне човникове сканування;
      * інакше -> PSO з відштовхуванням від сусідів.
    """
    name = 'PSO-LS'
    cv_flat, stag_k, stag_th, c3 = 0.35, 6, 0.3, 2.0

    def reset(self, env, rng):
        super().reset(env, rng)
        M = env.M
        self.mode = ['pso'] * M
        self.levy_tgt = np.full(M, -1)
        self.hist = [[] for _ in range(M)]
        self.heading = rng.choice([-1, 1], size=M)
        self.vdir = np.full(M, -1)
        self.sweep_wp = np.full(M, -1)

    def act(self, env):
        mp = env.mp
        F = env.gain_map()
        self.update_bests(env, F)
        Ffree = F[mp.free].mean() + 1e-15
        blur = uniform_filter(F.reshape(H, W), size=9, mode='constant').ravel()
        Fd = np.append(F, np.nan)
        out = env.pos.copy()
        for i in range(env.M):
            c = env.pos[i]
            self.hist[i] = (self.hist[i] + [F[c]])[-self.stag_k:]
            # --- режим Леві ---
            if self.mode[i] == 'levy':
                if c == self.levy_tgt[i]:
                    self.mode[i] = 'pso'
                else:
                    out[i] = step_toward(mp, c, self.levy_tgt[i], tie=F)
                    self.x[i] = RC[out[i]]
                    continue
            if len(self.hist[i]) == self.stag_k and np.mean(self.hist[i]) < self.stag_th * Ffree:
                ang = self.rng.uniform(0, 2 * np.pi, 8)
                L = np.clip(5 * (self.rng.pareto(1.0, 8) + 1), 5, 35)
                cand = [snap(mp, RC[c] + L[k] * np.array([np.sin(ang[k]), np.cos(ang[k])])) for k in range(8)]
                self.levy_tgt[i] = cand[int(np.argmax(blur[cand]))]
                self.mode[i] = 'levy'
                self.hist[i] = []
                out[i] = step_toward(mp, c, self.levy_tgt[i], tie=F)
                self.x[i] = RC[out[i]]
                continue
            w = mp.win[c]
            vals = Fd[w[w < N]]
            cv = vals.std() / (vals.mean() + 1e-15)
            if cv < self.cv_flat:
                # --- човникове сканування ---
                if self.sweep_wp[i] >= 0 and c != self.sweep_wp[i]:
                    out[i] = step_toward(mp, c, self.sweep_wp[i], tie=F)
                else:
                    self.sweep_wp[i] = -1
                    k = 4 if self.heading[i] > 0 else 3          # схід / захід
                    nxt = mp.nbr[c, k]
                    if nxt < 0:
                        r, cc = divmod(c, W)
                        if not (0 <= r + 3 * self.vdir[i] < H):
                            self.vdir[i] *= -1
                        self.sweep_wp[i] = snap(mp, (r + 3 * self.vdir[i], cc))
                        self.heading[i] *= -1
                        out[i] = step_toward(mp, c, self.sweep_wp[i], tie=F)
                    else:
                        out[i] = nxt
            else:
                # --- PSO з відштовхуванням ---
                tgt_pt = RC[self.pso_target(i, env)]
                rep = np.zeros(2)
                for j in range(env.M):
                    if j != i:
                        d = RC[c] - RC[env.pos[j]]
                        n2 = (d ** 2).sum()
                        if 0 < n2 < 16:
                            rep += d / n2
                        elif n2 == 0:
                            rep += self.rng.normal(0, 0.5, 2)
                tgt = snap(mp, tgt_pt + self.c3 * rep)
                out[i] = step_toward(mp, c, tgt, tie=F) if tgt != c else \
                    valid_nbrs(mp, c)[np.argmax(F[valid_nbrs(mp, c)])]
            self.x[i] = RC[out[i]]
        return out
