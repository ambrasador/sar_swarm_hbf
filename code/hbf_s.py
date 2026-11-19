# -*- coding: utf-8 -*-
"""
HBF-S — структурований варіант HBF (стаття 1, розширення): гіпотези — НЕ лише різні частки ρ рівномірної «решти
світу», а пари (ρ_k, альтернативна карта a_k):
        q^(k)_0(c) = min{0,95; N_v [ (1 − ρ_k) ψ(c) + ρ_k a_k(c) ]},
де a_k — рівномірний розподіл ('u') або одна з альтернативних карт планувальника mp.hyp_alts ('S').
Оновлення карт, правдоподібність (10), ваги (11), предиктивна карта (12) і рух — як у HBF (клас hbf.HBF
не змінюється; HBF-S успадковує observe/weights/belief/act).

Специфікація гіпотез spec — список пар (токен, ρ), токен: 'u' — рівномірна; 'S' — усі альтернативні карти
mp.hyp_alts з однаковим ρ; 'T' — справжня альтернатива (лише ORACLE-S, меж методу не належить).
Перший елемент — «карта правильна» (вага 1 − w_rest), решта розкладів гіпотез — порівну (вага w_rest).

Окремі випадки (перевіряє tests_s4.py): spec = [('u', 0,05)] — HEDAC* побітово; spec = [('u', ρ_1), ('u', ρ_2), ('u', ρ_3)]
з тими самими вагами — HBF з hbf.py побітово.
"""
import numpy as np

import baselines as B
from hbf import HBF
from env import N


def _expand(spec, mp):
    items = []
    for tok, rho in spec:
        if tok == 'S':
            items += [('a%d' % j, float(rho)) for j in range(len(mp.hyp_alts))]
        else:
            items.append((tok, float(rho)))
    return items


def _alt(tok, mp):
    if tok == 'u':
        return None
    if tok == 'T':
        return mp.alt_true
    return mp.hyp_alts[int(tok[1:])]


def _q0(mp, nv, rho, alt):
    if alt is None:
        return mp.q0(nv, row=float(rho))
    return np.minimum(0.95, nv * ((1 - rho) * mp.prior + rho * alt))


class HBFS(HBF):
    name = 'HBF-S'
    spec = (('u', 0.05), ('S', 0.6), ('u', 0.8))
    w_rest = 0.5
    prior_w = None
    beta = 1.0

    def reset(self, env, rng):
        B.Method.reset(self, env, rng)
        mp = env.mp
        nv = len(env.victims)
        items = _expand(self.spec, mp)
        self.items = items
        self.qm = [_q0(mp, nv, rho, _alt(tok, mp)) for tok, rho in items]
        K = len(items)
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

    def extra_metrics(self):
        w = self.weights()
        return {'w_final': [round(float(x), 4) for x in w]}


class OracleS(HBFS):
    """Еталон «знаємо істину»: одна гіпотеза — справжня суміш (1 − ρ) ψ + ρ a_true (ρ — справжня частка в екземплярі).
    Не є методом, придатним до застосування: показує запас для структурованих гіпотез."""
    name = 'ORACLE-S'
    spec = (('T', 0.0),)

    def reset(self, env, rng):
        rho = float(getattr(env, 'rho_true', 0.0))
        self.spec = (('T', rho),)
        super().reset(env, rng)
