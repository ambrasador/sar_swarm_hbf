# -*- coding: utf-8 -*-
"""Раннер експериментів на реальній місцевості (стаття 1): ставить сітку 50×50 (gridscale) ДО імпорту моделей,
потім — як run_s1.py (реєструє HBF, HBF-fix). Запуск: python run_s4_real.py --config configs_s4/real_main.json --workers 2"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['SAR_GRID'] = '50:s'
import gridscale
gridscale.install(50, scalable=True)
import run_s1  # noqa: E402,F401  (реєструє варіанти й розширює хеш коду)
import run_experiments as R  # noqa: E402
if __name__ == '__main__':
    sys.exit(R.main())
