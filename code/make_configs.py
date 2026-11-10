# -*- coding: utf-8 -*-
"""Стаття 1: з frozen.json створює variants_s1.py (параметри HBF записано літерально) і конфігурації
підтверджувальних експериментів configs_s1/*.json. Запускати один раз після make_protocol.py."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = json.load(open(os.path.join(HERE, 'frozen.json'), encoding='utf-8'))


def tup(d):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if v is not None}


p = tup(FZ['hbf'])
pfix = dict(p, beta=0.0)
sens = {f'HBFs:{k}': tup(v) for k, v in FZ['sens_hbf'].items()}
V = f'''# -*- coding: utf-8 -*-
"""
Стаття 1: реєстрація запропонованого методу HBF, його абляції та налаштувань для аналізу чутливості.
Параметри заморожено за протоколом PROTOCOL_S1.md (frozen.json; остаточна конфігурація — {FZ['hbf_name']} розроблення).
Файли моделей і variants.py не змінюються: модуль лише додає записи до variants.VARIANTS під час імпорту.
"""
import variants as V
import hbf

HBF_PARAMS = {p!r}
NEW = {{
    'HBF':     (hbf.HBF, dict(HBF_PARAMS), 'HBF', 'ієрархічне байєсівське потенціальне поле (заморожена конфігурація)'),
    'HBF-fix': (hbf.HBF, {pfix!r}, 'HBF', 'абляція: ваги гіпотез сталі (beta = 0)'),
}}
SENS = {sens!r}
for k, v in SENS.items():
    NEW[k] = (hbf.HBF, dict(v), 'HBF', 'аналіз чутливості до апріорного налаштування')
for k, v in NEW.items():
    V.VARIANTS.setdefault(k, v)
    if k not in V.VARIANT_ORDER:
        V.VARIANT_ORDER.append(k)
'''
open(os.path.join(HERE, 'variants_s1.py'), 'w', encoding='utf-8').write(V)

C = os.path.join(HERE, 'configs_s1')
os.makedirs(C, exist_ok=True)
eps, tag = FZ['eps_star'], FZ['eps_tag']
RH = ('00', '20', '40', '60', '80')
ARCH_RHO_EPS = {0.05, 0.10, 0.145, 0.20, 0.30}     # серія ρ для цих ε уже є (архів mis_r*, прогони статті A)
cfgs = [
    dict(name='s1_verify', scenarios=['S1', 'S2', 'S3'], maps=[11, 15], methods=['HEDAC*'],
         description='Стаття 1: контроль відтворюваності HEDAC* (має побітово збігтися з архівом conf_main)'),
    dict(name='s1_conf_hbf', scenarios=['S1', 'S2', 'S3'], maps=[11, 60], methods=['HBF', 'HBF-fix'],
         description='Стаття 1: підтвердження HBF і абляції HBF-fix на картах 11–60 (парно з conf_main)'),
    dict(name='s1_conf_eps', scenarios=['S1', 'S2', 'S3'], maps=[11, 60], methods=['HEDAC*'], row=eps,
         description=f'Стаття 1: конкурент HEDAC*-ε*, ε* = {eps}, карти 11–60'),
    dict(name='s1_conf_sens', scenarios=['S1', 'S2'], maps=[11, 60], methods=list(sens),
         description='Стаття 1: чутливість HBF до апріорного налаштування, S1–S2'),
]
for r in RH:
    rho = int(r) / 100
    cfgs.append(dict(name=f's1_mis_r{r}_hbf', scenarios=['S3'], maps=[11, 40], methods=['HBF', 'HBF-fix'],
                     rho_out=rho, description=f'Стаття 1: HBF і HBF-fix, rho_out = {rho}'))
    if eps not in ARCH_RHO_EPS:
        cfgs.append(dict(name=f's1_mis_r{r}_eps', scenarios=['S3'], maps=[11, 40], methods=['HEDAC*'],
                         rho_out=rho, row=eps, description=f'Стаття 1: HEDAC*-ε*, ε* = {eps}, rho_out = {rho}'))
    cfgs.append(dict(name=f's1_mis_r{r}_sens', scenarios=['S3'], maps=[11, 40], methods=list(sens),
                     rho_out=rho, description=f'Стаття 1: чутливість HBF, rho_out = {rho}'))
for e in FZ['sens_eps']:
    if abs(e - eps) < 1e-12 or abs(e - 0.05) < 1e-12:
        continue                                   # ε* і типове ε = 0,05 уже покрито
    et = f'{int(round(e * 100)):02d}'
    cfgs.append(dict(name=f's1_conf_e{et}', scenarios=['S1', 'S2'], maps=[11, 60], methods=['HEDAC*'], row=e,
                     description=f'Стаття 1: чутливість HEDAC*-ε, ε = {e}, S1–S2'))
    if e not in ARCH_RHO_EPS:
        for r in RH:
            rho = int(r) / 100
            cfgs.append(dict(name=f's1_mis_r{r}_e{et}', scenarios=['S3'], maps=[11, 40], methods=['HEDAC*'],
                             rho_out=rho, row=e, description=f'Стаття 1: чутливість HEDAC*-ε, ε = {e}, rho_out = {rho}'))
base = dict(draws=4, n_agents=5, T=300)
for c in cfgs:
    c = {**{k: c[k] for k in ('name', 'scenarios', 'maps')}, **base, **{k: v for k, v in c.items() if k not in ('name', 'scenarios', 'maps')}}
    json.dump(c, open(os.path.join(C, c['name'] + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(os.path.join(HERE, 'suite_s1.txt'), 'w').write('\n'.join(c['name'] for c in cfgs) + '\n')
print('ok', len(cfgs), [c['name'] for c in cfgs])
