# -*- coding: utf-8 -*-
"""
Телеметрія HBF (стаття 1): траєкторія апостеріорної оцінки частки постраждалих поза моделлю
ρ̂(t) = Σ_k π_k(t) ρ_k та ваг гіпотез на тих самих екземплярах, що й підтверджувальний прогін
(ті самі зерна; прогони детерміновані, тож SWS збігається з results/s1_*/runs.jsonl — перевіряється).

Запуск: python trace_s1.py --workers 2   → results/s1_trace/trace.json
"""
import argparse, json, os, sys, zlib
import multiprocessing as mp_
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiments as R  # noqa: E402
import variants_s1 as VS  # noqa: E402

RHOS = ('00', '20', '40', '60', '80')


def one(args):
    cond, sc, rho, ms, vd = args
    cfg = dict(name='trace', n_agents=5, T=300)
    if rho is not None:
        cfg['rho_out'] = rho
    mp = R.get_map(sc, ms, cfg)
    victims, flags = R.victims_for(cfg, mp, sc, ms, vd)
    inst = f'{sc}-m{ms}-v{vd}'
    env = R.make_env(cfg, mp, victims, zlib.crc32(inst.encode()), inst)
    cls, params = VS.NEW['HBF'][0], VS.NEW['HBF'][1]
    m = cls()
    for k, v in params.items():
        setattr(m, k, v)
    m.reset(env, np.random.default_rng(7 * ms + vd))
    rh = np.asarray(m.rhos, float)
    est, W = [float(m.weights() @ rh)], [m.weights().tolist()]
    while not env.done():
        env.step(m.act(env))
        m.observe(env)                       # ваги після спостережень цього кроку (на рух не впливає)
        w = m.weights()
        est.append(float(w @ rh))
        W.append(w.tolist())
    return dict(cond=cond, inst=inst, sws=float(env.metrics()[0]['sws']), est=est, w=W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    tasks = [('S1', 'S1', None, ms, vd) for ms in range(11, 61) for vd in range(4)]
    tasks += [('S2', 'S2', None, ms, vd) for ms in range(11, 61) for vd in range(4)]
    tasks += [(f'R{r}', 'S3', int(r) / 100, ms, vd) for r in RHOS for ms in range(11, 41) for vd in range(4)]
    with mp_.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=4)
    out = {}
    for r in res:
        out.setdefault(r['cond'], []).append(r)
    summ = {}
    for c, rs in out.items():
        E = np.array([r['est'] for r in rs])
        Wm = np.array([r['w'] for r in rs]).mean(0)
        summ[c] = dict(n=len(rs), est_mean=E.mean(0).round(5).tolist(), est_q25=np.percentile(E, 25, 0).round(5).tolist(),
                       est_q75=np.percentile(E, 75, 0).round(5).tolist(), w_mean=Wm.round(5).tolist(),
                       sws={r['inst']: r['sws'] for r in rs})
    d = os.path.join(HERE, 'results', 's1_trace')
    os.makedirs(d, exist_ok=True)
    json.dump(dict(rhos=list(VS.NEW['HBF'][1]['rhos']), summary=summ), open(os.path.join(d, 'trace.json'), 'w'))
    print('ok', {c: v['n'] for c, v in summ.items()})


if __name__ == '__main__':
    main()
