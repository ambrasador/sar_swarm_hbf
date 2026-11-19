# -*- coding: utf-8 -*-
"""Конфігурації розроблення S4 (лише карти 100–159; підтверджувальні карти 61–99 не використовуються)."""
import json, sys
def base_methods(eps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    m = {'HEDAC*': {'cls': 'HEDAC'}}
    for e in eps:
        m[f'HEDAC*|e{int(round(e*100)):02d}'] = {'cls': 'HEDAC', 'row': e}
    m['HBF-U'] = {'cls': 'HBF', 'params': {'rhos': [0.05, 0.3, 0.8], 'w_rest': 0.5, 'beta': 1.0}}
    m['BSS'] = {'cls': 'BSS'}
    m['ORACLE'] = {'cls': 'ORACLE'}
    return m
def hs(name_tail, rho_h, w, beta=1.0, uni=True):
    spec = [['u', 0.05], ['S', rho_h]] + ([['u', 0.8]] if uni else [])
    return {'cls': 'HBFS', 'params': {'spec': spec, 'w_rest': w, 'beta': beta}}
def hs_name(rho_h, w, beta, uni):
    return f'HS:r{int(rho_h*100)}-w{int(w*100)}-b{beta:g}-{"u" if uni else "n"}'
if __name__ == '__main__':
    conds = ['A0', 'A40', 'A80', 'B0', 'B40', 'B80']
    m = base_methods()
    for rho_h in (0.5, 0.8):
        for w in (0.5, 0.7):
            for uni in (True, False):
                m[hs_name(rho_h, w, 1.0, uni)] = hs('', rho_h, w, 1.0, uni)
    m[hs_name(0.8, 0.5, 2.0, True)] = hs('', 0.8, 0.5, 2.0, True)
    m[hs_name(0.5, 0.5, 2.0, True)] = hs('', 0.5, 0.5, 2.0, True)
    json.dump(dict(name='s4_dev_r1', conds=conds, maps=[100, 119], draws=2, methods=m,
                   description='S4, розроблення, раунд 1: карти 100–119, 2 розстановки'),
              open('configs_s4/dev_r1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(m), 'методів')
