# -*- coding: utf-8 -*-
"""
Розробка (раунд 4): варіант HBF з розподілом ролей між гіпотезами замість усереднення карт.
Кожен агент i має роль k_i і рухається за полем власної карти гіпотези φ_k = G(q^(k)·p_det);
кількість агентів у ролі k — найбільші залишки від M·π_k(t); при зміні кількостей роль змінюють
мінімальній кількості агентів (тим, у кого поле поточної ролі в їхній клітинці найменше).
Усі інші частини (K карт, правдоподібність, загартування, мікрокрок) — як у BMAField.
"""
import numpy as np
from bmaf import BMAField
from env import RC, valid_nbrs


def apportion(w, M):
    raw = w * M
    n = np.floor(raw).astype(int)
    rem = M - n.sum()
    if rem > 0:
        n[np.argsort(-(raw - n), kind='stable')[:rem]] += 1
    return n


class BMAFieldRoles(BMAField):
    name = 'BMA-FR'

    def reset(self, env, rng):
        super().reset(env, rng)
        n = apportion(self.weights(), env.M)
        self.role = np.repeat(np.arange(len(self.rhos)), n)

    def act(self, env):
        mp = env.mp
        self.observe(env)
        w = self.weights()
        n = apportion(w, env.M)
        phi = {}

        def field(k):
            if k not in phi:
                phi[k] = mp.G @ (self.qm[k] * mp.pdet).astype(np.float32)
            return phi[k]
        cur = np.bincount(self.role, minlength=len(self.rhos))
        while (cur != n).any():
            over = int(np.argmax(cur - n)); under = int(np.argmax(n - cur))
            cand = np.nonzero(self.role == over)[0]
            f = field(over)
            i = cand[np.argmin(f[env.pos[cand]])]
            self.role[i] = under
            cur[over] -= 1; cur[under] += 1
        out, chosen = env.pos.copy(), []
        for i in self.rng.permutation(env.M):
            ph = field(int(self.role[i]))
            c0 = env.pos[i]
            nb = valid_nbrs(mp, c0)
            vals = ph[nb].astype(float)
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
