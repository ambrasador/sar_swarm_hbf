# -*- coding: utf-8 -*-
"""
Стаття 1: реєстрація запропонованого методу HBF, його абляції та налаштувань для аналізу чутливості.
Параметри заморожено за протоколом PROTOCOL_S1.md (frozen.json; остаточна конфігурація — FG3-w50 розроблення).
Файли моделей і variants.py не змінюються: модуль лише додає записи до variants.VARIANTS під час імпорту.
"""
import variants as V
import hbf

HBF_PARAMS = {'rhos': (0.05, 0.3, 0.8), 'w_rest': 0.5, 'beta': 1.0}
NEW = {
    'HBF':     (hbf.HBF, dict(HBF_PARAMS), 'HBF', 'ієрархічне байєсівське потенціальне поле (заморожена конфігурація)'),
    'HBF-fix': (hbf.HBF, {'rhos': (0.05, 0.3, 0.8), 'w_rest': 0.5, 'beta': 0.0}, 'HBF', 'абляція: ваги гіпотез сталі (beta = 0)'),
}
SENS = {'HBFs:FG3-w30': {'rhos': (0.05, 0.3, 0.8), 'w_rest': 0.3}, 'HBFs:FG3-w70': {'rhos': (0.05, 0.3, 0.8), 'w_rest': 0.7}, 'HBFs:FG5-w50': {'rhos': (0.05, 0.2, 0.4, 0.6, 0.8), 'w_rest': 0.5}, 'HBFs:F2-w30': {'rhos': (0.05, 1.0), 'w_rest': 0.3}}
for k, v in SENS.items():
    NEW[k] = (hbf.HBF, dict(v), 'HBF', 'аналіз чутливості до апріорного налаштування')
for k, v in NEW.items():
    V.VARIANTS.setdefault(k, v)
    if k not in V.VARIANT_ORDER:
        V.VARIANT_ORDER.append(k)
