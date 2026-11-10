# -*- coding: utf-8 -*-
"""Обгортка над run_experiments.py для експериментів статті 1: реєструє HBF і його абляцію з variants_s1.py
та включає hbf.py і variants_s1.py у хеш коду експерименту. Сам раннер і файли моделей не змінюються.

Запуск:  python run_s1.py --config configs_s1/<назва>.json --workers 2
"""
import multiprocessing as mp_
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:                                            # Linux: fork — робочі процеси успадковують зареєстровані варіанти
    mp_.set_start_method('fork', force=True)
except ValueError:                              # Windows: spawn — дочірні процеси імпортують цей модуль,
    pass                                        # тож variants_s1 реєструється і в них
import variants_s1  # noqa: E402,F401
import run_experiments as R  # noqa: E402

for f in ('hbf.py', 'variants_s1.py'):
    if f not in R.EXT_FILES:
        R.EXT_FILES.append(f)
_orig = R.required_modules


def _req(cfg):
    mods = set(_orig(cfg))
    if any(m in variants_s1.NEW for m in R.method_list(cfg)):
        mods.update(('hbf.py', 'variants_s1.py'))
    return sorted(mods)


R.required_modules = _req

if __name__ == '__main__':
    sys.exit(R.main())
