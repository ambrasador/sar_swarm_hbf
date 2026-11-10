# -*- coding: utf-8 -*-
"""Усі конфігурації розроблення HBF (раунди 1–4) і конкурента HEDAC*-ε — єдиний реєстр (дослівно як у dev_r*.py)."""
import bmaf, bmaf_r

G5, G3 = (0.05, 0.2, 0.4, 0.6, 0.8), (0.05, 0.3, 0.8)
CFG = {
    'H-e05': {'rhos': (0.05,)}, 'H-e10': {'rhos': (0.10,)}, 'H-e145': {'rhos': (0.145,)}, 'H-e20': {'rhos': (0.20,)},
    'H-e30': {'rhos': (0.30,)}, 'H-e40': {'rhos': (0.40,)}, 'H-e50': {'rhos': (0.50,)}, 'H-e60': {'rhos': (0.60,)},
    'F2-w10': {'rhos': (0.05, 1.0), 'w_rest': 0.1}, 'F2-w30': {'rhos': (0.05, 1.0), 'w_rest': 0.3},
    'FG5-u': {'rhos': G5, 'prior_w': (1, 1, 1, 1, 1)}, 'FG5-u-b2': {'rhos': G5, 'prior_w': (1, 1, 1, 1, 1), 'beta': 2.0},
    'FG5-w50': {'rhos': G5, 'w_rest': 0.5}, 'FG5-w30': {'rhos': G5, 'w_rest': 0.3}, 'FG5-w70': {'rhos': G5, 'w_rest': 0.7},
    'FG5-w50-b2': {'rhos': G5, 'w_rest': 0.5, 'beta': 2.0}, 'FG5-w50-b05': {'rhos': G5, 'w_rest': 0.5, 'beta': 0.5},
    'FG6-w50': {'rhos': (0.05, 0.15, 0.3, 0.5, 0.7, 0.9), 'w_rest': 0.5},
    'FG3-w50': {'rhos': G3, 'w_rest': 0.5}, 'FG3-w30': {'rhos': G3, 'w_rest': 0.3}, 'FG3-w70': {'rhos': G3, 'w_rest': 0.7},
    'FG3-w50-b05': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.5}, 'FG3-w50-b2': {'rhos': G3, 'w_rest': 0.5, 'beta': 2.0},
    'FG3a-w50': {'rhos': (0.05, 0.25, 0.6), 'w_rest': 0.5}, 'FG3b-w50': {'rhos': (0.05, 0.4, 0.9), 'w_rest': 0.5},
    'FG3c-w50': {'rhos': (0.05, 0.3, 1.0), 'w_rest': 0.5},
    'FG2-w50': {'rhos': (0.05, 0.5), 'w_rest': 0.5}, 'FG2b-w50': {'rhos': (0.05, 0.8), 'w_rest': 0.5},
    'FG3-w50-b0': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.0},
    'FR3-w50': {'rhos': G3, 'w_rest': 0.5}, 'FR3-w50-b0': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.0},
    'FR2-w30': {'rhos': (0.05, 1.0), 'w_rest': 0.3}, 'FR5-w50': {'rhos': G5, 'w_rest': 0.5},
}
ROUND = {k: 1 for k in ['H-e05', 'H-e10', 'H-e145', 'H-e20', 'H-e30', 'H-e40', 'F2-w10', 'F2-w30', 'FG5-u', 'FG5-w50', 'FG5-u-b2']}
ROUND.update({k: 2 for k in ['FG5-w30', 'FG5-w70', 'FG5-w50-b2', 'FG5-w50-b05', 'FG6-w50', 'FG3-w50']})
ROUND.update({k: 4 for k in CFG if k.startswith('FR')})
ROUND.update({k: 3 for k in CFG if k not in ROUND})


def cls_of(name):
    return bmaf_r.BMAFieldRoles if name.startswith('FR') else bmaf.BMAField


def is_ablation(name):
    return CFG[name].get('beta', 1.0) == 0.0
