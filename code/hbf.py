# -*- coding: utf-8 -*-
"""
HBF — ієрархічне байєсівське потенціальне поле (стаття 1, листопад 2026).

Ідея. Апріорна карта ψ правильна лише для частини постраждалих; частка ρ постраждалих поза апріорною
моделлю («маса решти світу») невідома. HBF розглядає ρ як невідомий параметр ієрархічної моделі:
  • K гіпотез ρ_1 < … < ρ_K з апріорними вагами π_k;
  • для кожної гіпотези — власна байєсівська карта переконань
        q^(k)_0(c) = min{0,95; N_v[(1 − ρ_k) ψ(c) + ρ_k / |Ω|]},
    яку оновлюють усі огляди й виявлення (те саме правило, що й у HEDAC*/BSS);
  • апостеріорні ваги π_k(t) ∝ π_k · L_k(t)^β, де L_k — правдоподібність усієї історії оглядів
    (огляд без виявлення: 1 − q p; виявлення: q p), β — показник «загартування» (β = 1 — звичайний Байєс,
    β = 0 — ваги не навчаються, абляція HBF-fix);
  • апостеріорно-предиктивна карта q̄(t) = Σ_k π_k(t) q^(k)(t);
  • рух — потенціальне поле HEDAC*: φ = G (q̄ · p_det), той самий мікрокрок з інерцією μ і забороною
    сусідства клітинок, уже обраних іншими агентами на цьому кроці.

Окремі випадки (перевіряє tests_s1.py): K = 1, ρ_1 = 0,05 — HEDAC* (побітово); K = 1, ρ_1 = ε — HEDAC* з
рівномірною складовою ε (побітово з env row = ε).

Остаточні параметри (заморожено за протоколом PROTOCOL_S1.md до підтверджувального прогону) задає
variants_s1.py; типові значення класу нижче — лише для автономного використання.
Код дослівно повторює клас BMAField з dev/bmaf.py, на якому вели розробку (перейменовано лише клас).
"""
import numpy as np

import baselines as B
from env import N, RC, valid_nbrs


class HBF(B.Method):
    name = 'HBF'
    rhos = (0.05, 0.3, 0.8)   # гіпотези щодо частки постраждалих поза апріорною моделлю
    prior_w = None             # апріорні ваги гіпотез; None — 1 − w_rest першій, решта порівну
    w_rest = 0.5
    beta = 1.0                 # показник «загартування» правдоподібності (0 — ваги не навчаються)
    mu = 0.15                  # інерція мікрокроку (як у HEDAC*)

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
        """Байєсівське оновлення K карт і накопичення їхніх логарифмів правдоподібності."""
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
