import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета (на одному рівні вище)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev, bmaf
# Доповнення раунду 1 до всіх 20 карт підбору (112–119), щоб кожну конфігурацію раундів 1–3 оцінено на тих самих
# 560 екземплярах. Нових конфігурацій немає; відбір — лише на картах 100–119.
C = {
 'H-e10': {'rhos': (0.10,)}, 'H-e145': {'rhos': (0.145,)},
 'F2-w10': {'rhos': (0.05, 1.0), 'w_rest': 0.1}, 'F2-w30': {'rhos': (0.05, 1.0), 'w_rest': 0.3},
 'FG5-u': {'rhos': (0.05, 0.2, 0.4, 0.6, 0.8), 'prior_w': (1, 1, 1, 1, 1)},
 'FG5-u-b2': {'rhos': (0.05, 0.2, 0.4, 0.6, 0.8), 'prior_w': (1, 1, 1, 1, 1), 'beta': 2.0},
}
for k, v in C.items():
    dev.register(k, bmaf.BMAField, v)
if __name__ == '__main__':
    out = dev.run(list(C) + ['BSS'], maps=range(112, 120))
    json.dump(out, open('dev_r3b.json', 'w'))
    for n, r in dev.summary(out).items():
        print(f"{n:12s}", ' '.join(f"{k}:{v:.3f}" for k, v in r.items()))
