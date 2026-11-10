# -*- coding: utf-8 -*-
"""
Розробка покращеного BSS для статті 1 (листопад 2026).

BMAField — рій веде K гіпотез щодо маси «решти світу» ρ_k (частки постраждалих поза апріорною картою),
для кожної — власну байєсівську карту q^(k), що починається з q0 = min{0,95; N_v[(1−ρ_k)ψ + ρ_k/|Ω|]} і
оновлюється всіма оглядами й виявленнями; ваги гіпотез — апостеріорні (правдоподібність усіх спостережень,
за потреби «загартована» показником β). Рух — потенціальне поле HEDAC* над усередненою картою
q̄ = Σ π_k q^(k): φ = G (q̄ · p_det), той самий мікрокрок з інерцією й забороною сусідства обраних клітинок.

Окремі випадки: K = 1 (ρ = ε) — HEDAC* з рівномірною складовою ε (побітово, якщо ε = ROW_EPS);
K = 2 (ρ ∈ {0,05; 1}) — модельне усереднення BSS без ролей скептиків, але з навігацією полем.
"""
import numpy as np

import baselines as B
from env import N, RC, valid_nbrs


class BMAField(B.Method):
    name = 'BMA-F'
    rhos = (0.05, 1.0)       # гіпотези щодо маси «решти світу»
    prior_w = None            # апріорні ваги гіпотез; None — 1 − w_rest першій, решта порівну
    w_rest = 0.1
    beta = 1.0                # показник «загартування» правдоподібності
    mu = 0.15                 # інерція мікрокроку (як у HEDAC*)

    def reset(self, env, rng):
        super().reset(env, rng)
        nv = len(env.victims)
        self.qm = [env.mp.q0(nv, row=float(r)) for r in self.rhos]
        K = len(self.rhos)
        if self.prior_w is not None:
            w = np.asarray(self.prior_w, float)
        elif K == 1:
            w = np.ones(1)
        else:
            w = np.r_[1 - self.w_rest, np.full(K - 1, self.w_rest / (K - 1))]
        self.logw = np.log(w / w.sum())
        self.loglik = np.zeros(K)
        self.prev = np.zeros((env.M, 2))
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
            for m in range(len(self.qm)):
                q = self.qm[m][fi]
                lik = np.where(det, q * fp, 1 - q * fp)
                self.loglik[m] += np.log(np.maximum(lik, 1e-300)).sum()
                self.qm[m][fi] = np.where(det, 0.0, q * (1 - fp) / (1 - q * fp))
            for c in fi[det]:
                newly.discard(c)
        self.seen_t = env.t

    def weights(self):
        z = self.logw + self.beta * self.loglik
        w = np.exp(z - z.max())
        return w / w.sum()

    def belief(self):
        w = self.weights()
        q = w[0] * self.qm[0]
        for k in range(1, len(self.qm)):
            q = q + w[k] * self.qm[k]
        return q

    def act(self, env):
        mp = env.mp
        self.observe(env)
        qbar = self.belief()
        phi = mp.G @ (qbar * mp.pdet).astype(np.float32)
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
