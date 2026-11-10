import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета (на одному рівні вище)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev, bmaf
# Раунд 3 (останній раунд розробки; лише карти підбору 100–119). Після нього метод і ε конкурента заморожуються.
G3 = (0.05, 0.3, 0.8)
C = {
 'H-e50': {'rhos': (0.50,)}, 'H-e60': {'rhos': (0.60,)},
 'FG3-w30': {'rhos': G3, 'w_rest': 0.3}, 'FG3-w70': {'rhos': G3, 'w_rest': 0.7},
 'FG3-w50-b05': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.5}, 'FG3-w50-b2': {'rhos': G3, 'w_rest': 0.5, 'beta': 2.0},
 'FG3a-w50': {'rhos': (0.05, 0.25, 0.6), 'w_rest': 0.5}, 'FG3b-w50': {'rhos': (0.05, 0.4, 0.9), 'w_rest': 0.5},
 'FG3c-w50': {'rhos': (0.05, 0.3, 1.0), 'w_rest': 0.5},
 'FG2-w50': {'rhos': (0.05, 0.5), 'w_rest': 0.5}, 'FG2b-w50': {'rhos': (0.05, 0.8), 'w_rest': 0.5},
 'FG3-w50-b0': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.0},
}
for k, v in C.items():
    dev.register(k, bmaf.BMAField, v)
if __name__ == '__main__':
    out = dev.run(list(C), maps=range(100, 120))
    json.dump(out, open('dev_r3.json', 'w'))
    for n, r in dev.summary(out).items():
        print(f"{n:12s}", ' '.join(f"{k}:{v:.3f}" for k, v in r.items()))
