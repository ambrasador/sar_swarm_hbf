# -*- coding: utf-8 -*-
"""Перевірки коректності розширень статті 1 (S4, HBF-S). Запуск: python tests_s4.py"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as E, baselines as B, hbf as HB, hbf_s as HS, s4_env as S4, ext_env as X

ok = True
def check(name, cond):
    global ok
    print(('OK   ' if cond else 'FAIL '), name); ok &= bool(cond)

# 1. S4Map повторює карти S1/S2 побітово
for kind, sc in (('A', 'S1'), ('B', 'S2')):
    for seed in (11, 62, 120):
        a, b = S4.S4Map(kind, seed), E.MapInstance(sc, seed)
        check(f'S4Map {kind} seed {seed}: prior/free/terrain/base збігаються з {sc}',
              np.array_equal(a.prior, b.prior) and np.array_equal(a.free, b.free) and np.array_equal(a.terrain, b.terrain) and a.base == b.base)
        check(f'  alt_true та hyp_alts — розподіли на вільних клітинках', abs(a.alt_true.sum() - 1) < 1e-9 and all(abs(h.sum() - 1) < 1e-9 for h in a.hyp_alts)
              and a.alt_true[~a.free].sum() == 0)

def run(mp, method, victims, row=None, ms=5, vd=0, rho=0.0):
    env = X.SAREnvExt(mp, victims, 12345 + vd, row=row); env.rho_true = rho
    method.reset(env, np.random.default_rng(7 * ms + vd))
    while not env.done():
        env.step(method.act(env))
    return env.found_t.copy(), env.metrics()[0]['sws']

# 2. HBF-S з однією рівномірною гіпотезою ρ = 0,05 — HEDAC* побітово; ρ = 0,3 — HEDAC*-ε
for kind in 'AB':
    mp = S4.S4Map(kind, 64)
    v, fl = mp.sample_victims4(np.random.default_rng(3), 0.4)
    h = HS.HBFS(); h.spec = (('u', 0.05),)
    f1, s1 = run(mp, B.HEDAC(), v)
    f2, s2 = run(mp, h, v)
    check(f'{kind}: HBF-S[(u,0.05)] == HEDAC* (found_t, SWS)', np.array_equal(f1, f2) and abs(s1 - s2) < 1e-15)
    h = HS.HBFS(); h.spec = (('u', 0.3),)
    f1, s1 = run(mp, B.HEDAC(), v, row=0.3)
    f2, s2 = run(mp, h, v)
    check(f'{kind}: HBF-S[(u,0.3)] == HEDAC*(row=0.3)', np.array_equal(f1, f2) and abs(s1 - s2) < 1e-15)
    # 3. три рівномірні гіпотези — HBF із hbf.py
    hb = HB.HBF(); hb.rhos = (0.05, 0.3, 0.8); hb.w_rest = 0.5; hb.beta = 1.0
    hs = HS.HBFS(); hs.spec = (('u', 0.05), ('u', 0.3), ('u', 0.8)); hs.w_rest = 0.5; hs.beta = 1.0
    f1, s1 = run(mp, hb, v); f2, s2 = run(mp, hs, v)
    check(f'{kind}: HBF-S[3 рівномірні] == HBF', np.array_equal(f1, f2) and abs(s1 - s2) < 1e-15)

# 4. генератор постраждалих: ρ = 0 — як sample_victims; ρ > 0 — частка поза моделлю
mp = S4.S4Map('A', 70)
v0, f0 = mp.sample_victims4(np.random.default_rng(5), 0.0)
v1 = mp.sample_victims(np.random.default_rng(5))
check('sample_victims4(ρ=0) == sample_victims', np.array_equal(v0, v1) and f0.all())
fr = np.mean([1 - mp.sample_victims4(np.random.default_rng(s), 0.4)[1].mean() for s in range(400)])
check(f'частка поза моделлю при ρ = 0,4 ≈ 0,4 (виміряно {fr:.3f})', abs(fr - 0.4) < 0.04)

# 5. ORACLE-S: нормування карт і вага = 1
o = HS.OracleS(); env = X.SAREnvExt(mp, v1, 1); env.rho_true = 0.4; o.reset(env, np.random.default_rng(1))
check('ORACLE-S: одна гіпотеза, вага 1', len(o.qm) == 1 and abs(o.weights()[0] - 1) < 1e-12)
print('ВСІ ПЕРЕВІРКИ ПРОЙДЕНО' if ok else 'ЄВІДОМІ ПОМИЛКИ'); sys.exit(0 if ok else 1)
