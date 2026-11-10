import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # код пакета (на одному рівні вище)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dev, configs_dev as CD
# Етап 2 (dev/STAGE2_RULE.md): кандидати за J̄ на картах 100–119 → прогін на картах 120–159.
FILES = ['dev_r1.json', 'dev_r3b.json', 'dev_r2.json', 'dev_r3.json', 'dev_r4a.json']


def tuning_table():
    data = {}
    for f in FILES:
        for n, d in json.load(open(f)).items():
            for c, inst in d.items():
                data.setdefault(n, {}).setdefault(c, {}).update(inst)
    return {n: float(np.mean([np.mean(list(d[c].values())) for c in d])) for n, d in data.items()
            if len(d) == 7 and all(len(v) == 80 for v in d.values())}


if __name__ == '__main__':
    J = tuning_table()
    hb = sorted([k for k in J if k in CD.CFG and not k.startswith('H-e') and not CD.is_ablation(k)], key=lambda k: -J[k])[:5]
    ep = sorted([k for k in J if k.startswith('H-e')], key=lambda k: -J[k])[:3]
    json.dump(dict(hbf=hb, eps=ep, J_100_119={k: J[k] for k in hb + ep}), open('stage2_candidates.json', 'w'), indent=1)
    print('кандидати', hb, ep, flush=True)
    for k in hb + ep:
        dev.register(k, CD.cls_of(k), CD.CFG[k])
    out = dev.run(hb + ep, maps=range(120, 160))
    json.dump(out, open('dev_r4b.json', 'w'))
    for n, r in dev.summary(out).items():
        print(f"{n:12s}", ' '.join(f"{k}:{v:.3f}" for k, v in r.items()))
