# -*- coding: utf-8 -*-
"""Раннер S4-M (істина поза класом гіпотез): надбудова над run_s4.py без зміни його коду.
  python run_s4m.py --config configs_s4/conf_mis.json --workers 2"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import run_s4 as R
import s4m_env as M

R.CONDS.update({'M40': ('M', 0.4, 3), 'M80': ('M', 0.8, 3), 'N40': ('N', 0.4, 3), 'N80': ('N', 0.8, 3)})
R.CODE_FILES = R.CODE_FILES + ['s4m_env.py', 'run_s4m.py']
_get, _vic = R.get_map, R.victims_for


def get_map(kind, ms):
    if kind in M.BASE:
        key = (kind, ms)
        if key not in R._MAPS:
            if len(R._MAPS) >= 3:
                R._MAPS.pop(next(iter(R._MAPS)))
            R._MAPS[key] = M.S4MMap(kind, ms)
        return R._MAPS[key]
    return _get(kind, ms)


def victims_for(cond, mp, ms, vd):
    kind, rho, k = R.CONDS[cond]
    if kind in M.VOFF:
        rng = np.random.default_rng(1000 * ms + vd + M.VOFF[kind])
        v, fl = mp.sample_victims4(rng, rho, k)
        return v, fl, rho, k
    return _vic(cond, mp, ms, vd)


R.get_map, R.victims_for = get_map, victims_for
if __name__ == '__main__':
    R.main()
