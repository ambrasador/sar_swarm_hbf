# -*- coding: utf-8 -*-
"""
Раннер розширених експериментів статті 1 (S4: структурована помилка; серія N_v). Самостійний, env.py і
run_experiments.py не змінюються; зерна й порядок викликів — як у run_experiments.py (парний дизайн).

Конфіг (JSON): name, conds [A0, A40, ...], maps [lo, hi], draws, methods {назва: {cls, params, row}}, T, n_agents.
Журнал: results/<name>/runs.jsonl (дописується; прогін можна переривати й продовжувати).

  python run_s4.py --config configs_s4/<назва>.json --workers 2
"""
import argparse, hashlib, json, multiprocessing as mp_, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    mp_.set_start_method('fork', force=True)
except ValueError:
    pass

import env as E                                    # noqa: E402
import baselines as B                              # noqa: E402
import novel as NV                                 # noqa: E402
import ext_env as X                                # noqa: E402
import hbf as HB                                   # noqa: E402
import hbf_s as HS                                 # noqa: E402
import s4_env as S4                                # noqa: E402

CODE_FILES = ['env.py', 'baselines.py', 'novel.py', 'ext_env.py', 'hbf.py', 'hbf_s.py', 's4_env.py', 'run_s4.py']
CLASSES = {'HEDAC': B.HEDAC, 'BSS': NV.BSS, 'HBF': HB.HBF, 'HBFS': HS.HBFS, 'ORACLE': HS.OracleS}
# умови: назва -> (вид, ρ, N_v)
CONDS = {'A0': ('A', 0.0, 3), 'A40': ('A', 0.4, 3), 'A80': ('A', 0.8, 3),
         'B0': ('B', 0.0, 3), 'B40': ('B', 0.4, 3), 'B80': ('B', 0.8, 3),
         'U40n3': ('U', 0.4, 3), 'U40n8': ('U', 0.4, 8), 'U80n3': ('U', 0.8, 3), 'U80n8': ('U', 0.8, 8)}


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    return h.hexdigest()[:16]


_MAPS = {}


def get_map(kind, ms):
    key = (kind, ms)
    if key not in _MAPS:
        if len(_MAPS) >= 3:
            _MAPS.pop(next(iter(_MAPS)))
        _MAPS[key] = E.MapInstance('S3', ms) if kind == 'U' else S4.S4Map(kind, ms)
    return _MAPS[key]


def victims_for(cond, mp, ms, vd):
    kind, rho, k = CONDS[cond]
    if kind == 'U':
        rng = np.random.default_rng(1000 * ms + vd + 400000 + 10 * k)
        v, fl = X.sample_victims_flags(mp, rng, k=k, p_in=round(1.0 - rho, 12))
        return v, fl, rho, k
    rng = np.random.default_rng(1000 * ms + vd + S4.VOFF[kind])
    v, fl = mp.sample_victims4(rng, rho, k)
    return v, fl, rho, k


def make_method(spec):
    m = CLASSES[spec['cls']]()
    for k, v in spec.get('params', {}).items():
        if k == 'spec':
            v = tuple((str(a), float(b)) for a, b in v)
        elif isinstance(v, list):
            v = tuple(v)
        setattr(m, k, v)
    return m


def run_one(cfg, cond, ms, vd, mname, ctx):
    w0 = time.perf_counter()
    kind = CONDS[cond][0]
    mp = get_map(kind, ms)
    victims, flags, rho, k = victims_for(cond, mp, ms, vd)
    inst = f'{cond}-m{ms}-v{vd}'
    spec = cfg['methods'][mname]
    M, T = cfg.get('n_agents', 5), cfg.get('T', 300)
    seed = zlib.crc32(inst.encode())
    row = spec.get('row')
    env = X.SAREnvExt(mp, victims, seed, M=M, T=T, row=row, key=inst)
    env.rho_true = rho
    m = make_method(spec)
    m.reset(env, np.random.default_rng(7 * ms + vd))
    t_act = 0.0
    while not env.done():
        t0 = time.perf_counter()
        a = m.act(env)
        t_act += time.perf_counter() - t0
        env.step(a)
    met, cp = env.metrics()
    rec = dict(key=f'{cfg["name"]}|{cond}|{ms}|{vd}|{mname}', exp=cfg['name'], cond=cond, kind=kind, map=ms, draw=vd,
               instance=inst, method=mname, rho=rho, n_victims=k)
    rec.update({a: float(b) for a, b in met.items()})
    rec['found_t'] = [int(x) for x in env.found_t]
    rec['victims'] = [int(x) for x in victims]
    rec['in_model'] = [int(x) for x in flags]
    rec['ms_per_step'] = 1000 * t_act / env.T
    if hasattr(m, 'extra_metrics'):
        rec.update(m.extra_metrics())
    if hasattr(mp, 'alt_info'):
        rec['alt_info'] = mp.alt_info
    rec['code_hash'] = ctx
    rec['wall_s'] = round(time.perf_counter() - w0, 4)
    return rec


def task(args):
    cfg, cond, ms, vd, names, ctx = args
    return [run_one(cfg, cond, ms, vd, n, ctx) for n in names]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
        good = 0
        pos = 0
        while pos < len(data):
            nl = data.find(b'\n', pos)
            if nl < 0:
                break
            try:
                done.add(json.loads(data[pos:nl].decode('utf-8'))['key'])
                good = nl + 1
            except Exception:
                good = nl + 1
            pos = nl + 1
        if good < len(data):
            with open(path, 'r+b') as f:
                f.truncate(good)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding='utf-8'))
    out_dir = os.path.join(HERE, 'results', cfg['name'])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'runs.jsonl')
    done = load_done(path)
    ctx = code_hash()
    lo, hi = cfg['maps']
    tasks = []
    for cond in cfg['conds']:
        for ms in range(lo, hi + 1):
            for vd in range(cfg['draws']):
                names = [n for n in cfg['methods'] if f'{cfg["name"]}|{cond}|{ms}|{vd}|{n}' not in done]
                if names:
                    tasks.append((cfg, cond, ms, vd, names, ctx))
    total = len(cfg['conds']) * (hi - lo + 1) * cfg['draws'] * len(cfg['methods'])
    if a.limit:
        tasks = tasks[:a.limit]
    print(f'{cfg["name"]}: всього {total}, виконано {len(done)}, завдань {len(tasks)}, код {ctx}', flush=True)
    t0, n = time.time(), 0
    with open(path, 'ab') as f, mp_.Pool(a.workers) as pool:
        for recs in pool.imap_unordered(task, tasks):
            for r in recs:
                f.write((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8'))
            f.flush()
            os.fsync(f.fileno())
            n += 1
            if n % 20 == 0:
                print(f'  {n}/{len(tasks)} завдань, {time.time() - t0:.0f} с', flush=True)
    print('готово', flush=True)


if __name__ == '__main__':
    main()
