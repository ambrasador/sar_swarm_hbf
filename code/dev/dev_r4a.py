import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета (на одному рівні вище)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev, bmaf_r
# Раунд 4а (карти підбору 100–119): розподіл ролей між гіпотезами замість усереднення карт.
G3, G5 = (0.05, 0.3, 0.8), (0.05, 0.2, 0.4, 0.6, 0.8)
C = {
 'FR3-w50': {'rhos': G3, 'w_rest': 0.5}, 'FR3-w50-b0': {'rhos': G3, 'w_rest': 0.5, 'beta': 0.0},
 'FR2-w30': {'rhos': (0.05, 1.0), 'w_rest': 0.3}, 'FR5-w50': {'rhos': G5, 'w_rest': 0.5},
}
for k, v in C.items():
    dev.register(k, bmaf_r.BMAFieldRoles, v)
if __name__ == '__main__':
    out = dev.run(list(C), maps=range(100, 120))
    json.dump(out, open('dev_r4a.json', 'w'))
    for n, r in dev.summary(out).items():
        print(f"{n:12s}", ' '.join(f"{k}:{v:.3f}" for k, v in r.items()))
