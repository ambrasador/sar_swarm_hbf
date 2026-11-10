# -*- coding: utf-8 -*-
"""Швидкий стенд розробки: лише карти підбору (100–119), ті самі зерна, що й у раннері."""
import json, sys, time, zlib
import multiprocessing as mp_
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants as V
import run_experiments as R
import bmaf

COND = [('S1', None), ('S2', None), ('S3', 0.0), ('S3', 0.2), ('S3', 0.4), ('S3', 0.6), ('S3', 0.8)]


def register(name, cls, params):
    V.VARIANTS[name] = (cls, params, 'BSS', 'розробка')


def one(args):
    name, sc, rho, ms, vd = args
    cfg = dict(name='dev', n_agents=5, T=300)
    if rho is not None:
        cfg['rho_out'] = rho
    mp = R.get_map(sc, ms, cfg)
    victims, flags = R.victims_for(cfg, mp, sc, ms, vd)
    inst = f'{sc}-m{ms}-v{vd}'
    env = R.make_env(cfg, mp, victims, zlib.crc32(inst.encode()), inst)
    m, _ = R.make_method(cfg, name, {})
    m.reset(env, np.random.default_rng(7 * ms + vd))
    while not env.done():
        env.step(m.act(env))
    return name, sc, rho, ms, vd, float(env.metrics()[0]['sws'])


def run(names, maps=range(100, 112), cond=COND, workers=2):
    tasks = [(n, sc, rho, ms, vd) for (sc, rho) in cond for ms in maps for vd in range(4) for n in names]
    tasks.sort(key=lambda t: (t[1], t[2] if t[2] is not None else -1, t[3]))
    with mp_.get_context('fork').Pool(workers) as pool:
        res = pool.map(one, tasks, chunksize=8)
    out = {}
    for n, sc, rho, ms, vd, s in res:
        out.setdefault(n, {}).setdefault(f'{sc}|{rho}', {})[f'{ms}-{vd}'] = s
    return out


def summary(out):
    rows = {}
    for n, d in out.items():
        per = {k: float(np.mean(list(v.values()))) for k, v in d.items()}
        rows[n] = dict(per, mean=float(np.mean(list(per.values()))))
    return rows
