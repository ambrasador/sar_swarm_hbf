# -*- coding: utf-8 -*-
"""
Статья 1, розширення: аналіз підтверджувальних прогонів за PROTOCOL_S4.md (гіпотези зафіксовано до прогону).

Джерела: журнали runs.jsonl (формат run_s4.py або run_experiments.py). Умова — префікс ідентифікатора екземпляра
(A0, A40, …, S1, S2, S3). Одиниця незалежності — карта (група, номер) — для S4: група = вид (A | B);
для реальної місцевості: група = сценарій.

Первинна кінцева точка для пари (a, b): D = (1/|C|) Σ_умов середня парна різниця SWS; рандомізаційний тест знаків
(100 000 перестановок, зерно 1) і стратифікований кластерний бутстреп (10 000, зерно 1) на рівні карт.
Запуск: python3 analysis_s4.py <конфіг-аналізу json> <вихід json>
"""
import json, os, sys
import numpy as np
from scipy import stats

CFG = json.load(open(sys.argv[1], encoding='utf-8'))
OUT = sys.argv[2]
DELTA = 0.02
HERE = os.path.dirname(os.path.abspath(__file__))


def load(exps):
    d = {}
    for e in exps:
        p = os.path.join(HERE, 'results', e, 'runs.jsonl')
        for line in open(p, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                d[(r['exp'] + ':' + r['method'], r['instance'])] = r
    return d


DATA = load(CFG['exps'])
METH = CFG['methods']            # метка -> назва методу в журналі
GROUP = CFG['groups']            # умова -> група (вид)


def parse(i):
    c, rest = i.split('-m')
    m, v = rest.split('-v')
    return c, int(m), int(v)


INST = {}
for (m, i) in DATA:
    c, ms, vd = parse(i)
    INST.setdefault(c, set()).add(i)
INST = {c: sorted(v, key=parse) for c, v in INST.items()}


def vec(c, label, tau=None):
    name = METH[label]
    out = []
    for i in INST[c]:
        r = DATA[(name, i)]
        if tau is None:
            out.append(r['sws'])
        else:
            ft = np.asarray(r['found_t'], float); ok = ft >= 0
            out.append(float(np.exp(-(ft[ok] / tau) ** 1.5).sum() / len(ft)))
    return np.array(out)


def map_means(c, x):
    ii = INST[c]
    ms = sorted({parse(i)[1] for i in ii})
    return ms, np.array([np.mean([x[k] for k, i in enumerate(ii) if parse(i)[1] == mm]) for mm in ms])


def holm(ps):
    ps = np.asarray(ps, float); adj, run = np.empty_like(ps), 0.0
    for k, i in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def boot(dif, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    m = dif[rng.integers(0, len(dif), size=(n, len(dif)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wil(x, y):
    return 1.0 if np.allclose(x - y, 0) else float(stats.wilcoxon(x, y, zero_method='zsplit').pvalue)


def a12(x, y):
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def noninf(dif, delta=DELTA):
    n, se = len(dif), dif.std(ddof=1) / np.sqrt(len(dif))
    return float(1 - stats.t.cdf((dif.mean() + delta) / se, n - 1))


def primary(a, b, conds, tau=None, nperm=100000, nboot=10000, seed=1):
    units = {}
    for c in conds:
        ms, md = map_means(c, vec(c, a, tau) - vec(c, b, tau))
        g = GROUP[c]
        for mm, v in zip(ms, md):
            units.setdefault(g, {}).setdefault(mm, 0.0)
            units[g][mm] += v
    ng = {g: len([c for c in conds if GROUP[c] == g]) for g in units}
    contrib = {g: np.array([units[g][k] for k in sorted(units[g])]) / (len(units[g]) * len(conds)) for g in units}
    allc = np.concatenate([contrib[g] for g in sorted(contrib)])
    D = float(allc.sum())
    rng = np.random.default_rng(seed)
    perm = (rng.choice([-1.0, 1.0], size=(nperm, len(allc))) * allc).sum(1)
    p = float((np.abs(perm) >= abs(D) - 1e-15).mean())
    rng = np.random.default_rng(seed)
    bs = np.zeros(nboot)
    for g in sorted(contrib):
        v = contrib[g]
        bs += v[rng.integers(0, len(v), size=(nboot, len(v)))].sum(1)
    return dict(D=D, p_perm=p, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_units={g: int(len(contrib[g])) for g in sorted(contrib)})


def cond_cmp(a, b, conds):
    out = {}
    for c in conds:
        x, y = vec(c, a), vec(c, b)
        d = x - y
        _, mx = map_means(c, x); _, my = map_means(c, y)
        out[c] = dict(n=len(d), a=float(x.mean()), b=float(y.mean()), diff=float(d.mean()), ci=boot(d, 5000),
                      p=wil(x, y), p_map=wil(mx, my), a12=a12(x, y), p_noninf=noninf(d))
    return out


R = dict(config=CFG)
conds = CFG['conds']
R['means'] = {c: {l: float(vec(c, l).mean()) for l in METH} for c in conds}
R['avg'] = {l: float(np.mean([R['means'][c][l] for c in conds])) for l in METH}
R['primary'] = {}
fam = CFG['family']                 # список пар [a, b] для поправки Холма
for a, b in CFG['pairs']:
    R['primary'][f'{a}|{b}'] = primary(a, b, conds)
ps = [R['primary'][f'{a}|{b}']['p_perm'] for a, b in fam]
for (a, b), ph in zip(fam, holm(ps) if ps else []):
    R['primary'][f'{a}|{b}']['p_holm'] = float(ph)
R['cond'] = {f'{a}|{b}': cond_cmp(a, b, conds) for a, b in CFG['pairs']}
if CFG.get('cost'):                 # не-гіршість у умовах правильної карти
    a, b = CFG['cost']['pair']
    R['cost'] = {c: dict(diff=R['cond'][f'{a}|{b}'][c]['diff'], ci=R['cond'][f'{a}|{b}'][c]['ci'],
                         p_noninf=R['cond'][f'{a}|{b}'][c]['p_noninf']) for c in CFG['cost']['conds']}
R['tau'] = {}
for tau in (120.0, 240.0):
    R['tau'][str(int(tau))] = {f'{a}|{b}': primary(a, b, conds, tau=tau, nperm=20000, nboot=3000) for a, b in CFG['pairs']}
if CFG.get('oracle'):
    o, a_, b_ = CFG['oracle']['oracle'], CFG['oracle']['method'], CFG['oracle']['ref']
    R['oracle_gap'] = {c: dict(oracle=R['means'][c][o], method=R['means'][c][a_], ref=R['means'][c][b_]) for c in conds}
R['ms_per_step'] = {l: float(np.median([DATA[(METH[l], i)]['ms_per_step'] for c in conds for i in INST[c]])) for l in METH}
if CFG.get('nv'):                   # різниця виграшів між N_v = 8 і N_v = 3 (парні карти)
    nv = CFG['nv']
    out = {}
    for (c8, c3) in nv['pairs']:
        for (a, b) in nv['gains']:
            def gm(c):
                ms, md = map_means(c, vec(c, a) - vec(c, b)); return dict(zip(ms, md))
            g8, g3 = gm(c8), gm(c3)
            ks = sorted(set(g8) & set(g3))
            d = np.array([g8[k] - g3[k] for k in ks])
            rng = np.random.default_rng(1)
            perm = (rng.choice([-1.0, 1.0], size=(100000, len(d))) * d).mean(1)
            out[f'{c8}-{c3}|{a}|{b}'] = dict(gain8=float(np.mean([g8[k] for k in ks])), gain3=float(np.mean([g3[k] for k in ks])),
                                             diff=float(d.mean()), ci=boot(d), p=float((np.abs(perm) >= abs(d.mean()) - 1e-15).mean()), n=len(d))
    for (a, b) in nv['gains']:
        pm = []
        for (c8, c3) in nv['pairs']:
            def gm(c):
                ms, md = map_means(c, vec(c, a) - vec(c, b)); return dict(zip(ms, md))
            g8, g3 = gm(c8), gm(c3)
            pm.append({k: g8[k] - g3[k] for k in set(g8) & set(g3)})
        ks = sorted(set.intersection(*[set(p) for p in pm]))
        d = np.array([np.mean([p[k] for p in pm]) for k in ks])
        rng = np.random.default_rng(1)
        perm = (rng.choice([-1.0, 1.0], size=(100000, len(d))) * d).mean(1)
        out[f'pooled|{a}|{b}'] = dict(diff=float(d.mean()), ci=boot(d), p=float((np.abs(perm) >= abs(d.mean()) - 1e-15).mean()), n=len(d))
    R['nv'] = out
json.dump(R, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('збережено', OUT)
