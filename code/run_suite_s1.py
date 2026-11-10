# -*- coding: utf-8 -*-
"""Послідовний запуск усіх підтверджувальних експериментів статті 1 (список — suite_s1.txt).
Запуск: python run_suite_s1.py --workers 2   (повторний запуск продовжує незавершені експерименти)"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
ap.add_argument('--only', default=None, help='запустити лише експерименти, назва яких містить цей рядок')
a = ap.parse_args()
names = [l.strip() for l in open(os.path.join(HERE, 'suite_s1.txt'), encoding='utf-8') if l.strip()]
for n in names:
    if a.only and a.only not in n:
        continue
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, 'run_s1.py'), '--config',
                        os.path.join(HERE, 'configs_s1', n + '.json'), '--workers', str(a.workers)])
    print(f'{n}: код {r.returncode}, {time.time() - t:.0f} с', flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
