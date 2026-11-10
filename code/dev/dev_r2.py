import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета (на одному рівні вище)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev, bmaf
G5 = (0.05, 0.2, 0.4, 0.6, 0.8)
C = {
 'H-e05': {'rhos': (0.05,)}, 'H-e20': {'rhos': (0.20,)}, 'H-e30': {'rhos': (0.30,)}, 'H-e40': {'rhos': (0.40,)},
 'FG5-w50': {'rhos': G5, 'w_rest': 0.5}, 'FG5-w30': {'rhos': G5, 'w_rest': 0.3}, 'FG5-w70': {'rhos': G5, 'w_rest': 0.7},
 'FG5-w50-b2': {'rhos': G5, 'w_rest': 0.5, 'beta': 2.0}, 'FG5-w50-b05': {'rhos': G5, 'w_rest': 0.5, 'beta': 0.5},
 'FG6-w50': {'rhos': (0.05, 0.15, 0.3, 0.5, 0.7, 0.9), 'w_rest': 0.5},
 'FG3-w50': {'rhos': (0.05, 0.3, 0.8), 'w_rest': 0.5},
}
for k, v in C.items():
    dev.register(k, bmaf.BMAField, v)
if __name__ == '__main__':
    out = dev.run(list(C), maps=range(100, 120))
    json.dump(out, open('dev_r2.json', 'w'))
    for n, r in dev.summary(out).items():
        print(f"{n:12s}", ' '.join(f"{k}:{v:.3f}" for k, v in r.items()))
