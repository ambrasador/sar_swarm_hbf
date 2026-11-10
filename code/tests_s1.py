# -*- coding: utf-8 -*-
"""Перевірки коректності HBF (стаття 1). Запуск: python tests_s1.py
1) HBF з K = 1, ρ_1 = 0,05 побітово відтворює HEDAC* (SWS і моменти виявлення);
2) HBF з K = 1, ρ_1 = ε побітово відтворює HEDAC* у середовищі з рівномірною складовою ε (row = ε);
3) остаточна конфігурація HBF з variants_s1.py побітово збігається з класом BMAField (dev/bmaf.py),
   на якому вели розробку;
4) ваги гіпотез — розподіл імовірностей; за β = 0 вони дорівнюють апріорним на кожному кроці."""
import os, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'dev'))
import run_experiments as R  # noqa: E402
import baselines as B  # noqa: E402
import hbf  # noqa: E402
import variants_s1 as VS  # noqa: E402


def run(cls, params, sc, ms, vd, cfg_extra=None, hook=None):
    cfg = dict(name='test', n_agents=5, T=300, **(cfg_extra or {}))
    mp = R.get_map(sc, ms, cfg)
    victims, _ = R.victims_for(cfg, mp, sc, ms, vd)
    inst = f'{sc}-m{ms}-v{vd}'
    env = R.make_env(cfg, mp, victims, zlib.crc32(inst.encode()), inst)
    m = cls()
    for k, v in params.items():
        setattr(m, k, v)
    m.reset(env, np.random.default_rng(7 * ms + vd))
    while not env.done():
        env.step(m.act(env))
        if hook:
            hook(m)
    mt = env.metrics()[0]
    return float(mt['sws']), list(env.found_t)


CASES = [('S1', 11, 0), ('S2', 23, 1), ('S3', 37, 2)]
ok = True
for sc, ms, vd in CASES:
    a = run(B.HEDAC, {}, sc, ms, vd)
    b = run(hbf.HBF, dict(rhos=(0.05,)), sc, ms, vd)
    print('1)', sc, ms, vd, a[0], b[0], a == b); ok &= a == b
    a = run(B.HEDAC, {}, sc, ms, vd, dict(row=0.4, rho_out=0.4))
    b = run(hbf.HBF, dict(rhos=(0.4,)), sc, ms, vd, dict(row=0.4, rho_out=0.4))
    print('2)', sc, ms, vd, a[0], b[0], a == b); ok &= a == b
try:
    import bmaf
    cls, params = VS.NEW['HBF'][0], VS.NEW['HBF'][1]
    for sc, ms, vd in CASES:
        a = run(cls, params, sc, ms, vd)
        b = run(bmaf.BMAField, params, sc, ms, vd)
        print('3)', sc, ms, vd, a[0], b[0], a == b); ok &= a == b
except ImportError:
    print('3) dev/bmaf.py відсутній — пропущено')
ws = []
run(hbf.HBF, dict(VS.NEW['HBF-fix'][1]), 'S3', 12, 0, dict(rho_out=0.6), hook=lambda m: ws.append(m.weights()))
w0 = ws[0]
c4 = all(np.allclose(w, w0) for w in ws) and abs(w0.sum() - 1) < 1e-12
print('4) β = 0: ваги сталі', c4, np.round(w0, 4)); ok &= c4
print('УСІ ПЕРЕВІРКИ ПРОЙДЕНО' if ok else 'Є РОЗБІЖНОСТІ')
sys.exit(0 if ok else 1)
