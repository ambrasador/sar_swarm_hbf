# -*- coding: utf-8 -*-
"""
Стаття 1: аналіз підтверджувального прогону за протоколом PROTOCOL_S1.md (гіпотези зафіксовано до прогону).

Джерела (усі — журнали runs.jsonl раннера run_experiments.py):
  ARCH  — архів ПК sar_swarm v2 (conf_main, mis_r00…r80): HEDAC* (ε = 0,05), BSS та інші базові методи;
  AEXP  — прогони статті A (a_mis_rXX_eYY): HEDAC* з ε ∈ {0,10; 0,145; 0,20; 0,30} у серії ρ;
  NEW   — прогони статті 1 (s1_*): HBF, HBF-fix, HEDAC*-ε*, сімейства чутливості, контроль відтворюваності.

Умови (як у критерії розроблення): S1, S2 (карти 11–60, по 200 екземплярів), S3 з ρ ∈ {0; 0,2; 0,4; 0,6; 0,8}
(карти 11–40, по 120). Додатково: S3c — S3 за замовчуванням (ρ = 0,4), карти 11–60.

Первинна кінцева точка: D = (1/7) Σ_умов середня парна різниця SWS; одиниця незалежності — карта сценарію
(S1: 50, S2: 50, S3: 30, внесок карти S3 — сума за п'ятьма ρ); двобічний рандомізаційний тест знаків
(100 000 перестановок, зерно 1); 95 % ДІ — стратифікований кластерний бутстреп карт (10 000, зерно 1).

Запуск: python3 -I analysis_s1.py <ARCH> <AEXP> <NEW> frozen.json <вихід json>
"""
import json, os, sys
import numpy as np
from scipy import stats

ARCH, AEXP, NEW, FROZEN, OUT = sys.argv[1:6]
FZ = json.load(open(FROZEN, encoding='utf-8'))
EPS = FZ['eps_star']
RHOS = ('00', '20', '40', '60', '80')
CONDS = ['S1', 'S2'] + [f'R{r}' for r in RHOS]
DELTA = 0.02
TAUS = (120.0, 180.0, 240.0)
KAPPA = 1.5
_cache = {}


def load(root, exp):
    p = os.path.join(root, exp, 'runs.jsonl')
    if p not in _cache:
        d = {}
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d[(r['method'], r['instance'])] = r
        _cache[p] = d
    return _cache[p]


def inst_key(s):
    sc, rest = s.split('-m')
    m, v = rest.split('-v')
    return (sc, int(m), int(v))


def etag(e):
    return f'{int(round(e * 100)):02d}' if abs(e * 100 - round(e * 100)) < 1e-9 else str(e).replace('0.', '')


data = {c: {} for c in CONDS + ['S3c']}


def put(cond, recs, method, label, scenario=None):
    for (m, i), r in recs.items():
        if m == method and (scenario is None or r['scenario'] == scenario):
            data[cond].setdefault(label, {})[i] = r


SC = (('S1', 'S1'), ('S2', 'S2'), ('S3', 'S3c'))
# --- архів
conf = load(ARCH, 'conf_main')
for sc, cond in SC:
    for m in ('HEDAC*', 'BSS', 'GREEDY', 'SIFD', 'PAS'):
        put(cond, conf, m, m, sc)
for r in RHOS:
    for m in ('HEDAC*', 'BSS', 'GREEDY', 'SIFD', 'PAS'):
        put(f'R{r}', load(ARCH, f'mis_r{r}'), m, m)
    for e in ('10', '145', '20', '30'):
        put(f'R{r}', load(AEXP, f'a_mis_r{r}_e{e}'), 'HEDAC*', f'HEDAC*|e{e}')
# --- нові прогони
hb, ep = load(NEW, 's1_conf_hbf'), load(NEW, 's1_conf_eps')
for sc, cond in SC:
    put(cond, hb, 'HBF', 'HBF', sc); put(cond, hb, 'HBF-fix', 'HBF-fix', sc)
    put(cond, ep, 'HEDAC*', 'HEDAC*-eps', sc)
sens_names = [f'HBFs:{k}' for k in FZ['sens_hbf']]
sc_ = load(NEW, 's1_conf_sens')
for sc, cond in SC[:2]:
    for m in sens_names:
        put(cond, sc_, m, m, sc)
for r in RHOS:
    rec = load(NEW, f's1_mis_r{r}_hbf')
    put(f'R{r}', rec, 'HBF', 'HBF'); put(f'R{r}', rec, 'HBF-fix', 'HBF-fix')
    if os.path.isdir(os.path.join(NEW, f's1_mis_r{r}_eps')):
        put(f'R{r}', load(NEW, f's1_mis_r{r}_eps'), 'HEDAC*', 'HEDAC*-eps')
    else:
        put(f'R{r}', load(AEXP, f'a_mis_r{r}_e{FZ["eps_tag"]}'), 'HEDAC*', 'HEDAC*-eps')
    for m in sens_names:
        put(f'R{r}', load(NEW, f's1_mis_r{r}_sens'), m, m)
# --- сімейство ε (аналіз чутливості): мітки HEDAC*|eXX у всіх 7 умовах
for e in FZ['sens_eps']:
    lab = f'HEDAC*|e{etag(e)}'
    for sc, cond in SC[:2]:
        if abs(e - 0.05) < 1e-12:
            put(cond, conf, 'HEDAC*', lab, sc)
        elif abs(e - EPS) < 1e-12:
            put(cond, ep, 'HEDAC*', lab, sc)
        else:
            put(cond, load(NEW, f's1_conf_e{etag(e)}'), 'HEDAC*', lab, sc)
    for r in RHOS:
        if abs(e - 0.05) < 1e-12:
            put(f'R{r}', load(ARCH, f'mis_r{r}'), 'HEDAC*', lab)
        elif abs(e - EPS) < 1e-12:
            data[f'R{r}'][lab] = data[f'R{r}']['HEDAC*-eps']
        elif e in (0.10, 0.20, 0.30):
            pass                                        # уже завантажено з AEXP під цією міткою
        else:
            put(f'R{r}', load(NEW, f's1_mis_r{r}_e{etag(e)}'), 'HEDAC*', lab)

N_EXP = dict(S1=200, S2=200, S3c=200, **{f'R{r}': 120 for r in RHOS})
for c, dm in data.items():
    for m, d in dm.items():
        assert len(d) == N_EXP[c], (c, m, len(d))
    assert len({frozenset(d) for d in dm.values()}) == 1, c

MAIN = ['HEDAC*', 'HEDAC*-eps', 'BSS', 'HBF-fix', 'HBF']


def insts(c):
    return sorted(next(iter(data[c].values())), key=inst_key)


def sws_of(r, tau=180.0, kappa=KAPPA):
    ft = np.asarray(r['found_t'], float)
    ok = ft >= 0
    return float((np.exp(-(ft[ok] / tau) ** kappa)).sum() / len(ft))


err = max(abs(sws_of(r) - r['sws']) for c in data for m in data[c] for r in data[c][m].values())


def vec(c, m, tau=None):
    return np.array([data[c][m][i]['sws'] if tau is None else sws_of(data[c][m][i], tau) for i in insts(c)])


def map_of(i):
    return i.rsplit('-v', 1)[0]


def wil(x, y):
    if np.allclose(x - y, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method='zsplit').pvalue)


def boot(dif, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dif), size=(n, len(dif)))
    m = dif[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    ps = np.asarray(ps, float)
    adj, run = np.empty_like(ps), 0.0
    for k, i in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def map_means(c, x):
    ii = insts(c)
    ms = sorted({map_of(i) for i in ii}, key=lambda s: inst_key(s + '-v0'))
    return ms, np.array([np.mean([x[k] for k, i in enumerate(ii) if map_of(i) == mm]) for mm in ms])


def noninf(dif, delta=DELTA):
    n, se = len(dif), dif.std(ddof=1) / np.sqrt(len(dif))
    return float(1 - stats.t.cdf((dif.mean() + delta) / se, n - 1))


def primary(a, b, tau=None, nperm=100000, nboot=10000, seed=1):
    """D = середнє за 7 умовами парних різниць a − b; тест знаків і бутстреп на рівні карт сценарію."""
    units = {'S1': {}, 'S2': {}, 'S3': {}}
    for c in CONDS:
        ms, md = map_means(c, vec(c, a, tau) - vec(c, b, tau))
        st = 'S3' if c.startswith('R') else c
        for mm, v in zip(ms, md):
            units[st][mm] = units[st].get(mm, 0.0) + v / (len(ms) * len(CONDS))
    contrib = {s: np.array([u[k] for k in sorted(u)]) for s, u in units.items()}
    allc = np.concatenate([contrib[s] for s in ('S1', 'S2', 'S3')])
    D = float(allc.sum())
    rng = np.random.default_rng(seed)
    perm = (rng.choice([-1.0, 1.0], size=(nperm, len(allc))) * allc).sum(1)
    p = float((np.abs(perm) >= abs(D) - 1e-15).mean())
    rng = np.random.default_rng(seed)
    bs = np.zeros(nboot)
    for s in ('S1', 'S2', 'S3'):
        v = contrib[s]
        bs += v[rng.integers(0, len(v), size=(nboot, len(v)))].sum(1)
    return dict(D=D, p_perm=p, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_units={s: int(len(contrib[s])) for s in contrib})


def cond_cmp(a, b, conds):
    out = {}
    for c in conds:
        x, y = vec(c, a), vec(c, b)
        d = x - y
        _, mx = map_means(c, x)
        _, my = map_means(c, y)
        out[c] = dict(n=len(d), a=float(x.mean()), b=float(y.mean()), diff=float(d.mean()), ci=boot(d),
                      p=wil(x, y), p_map=wil(mx, my), a12=a12(x, y), p_noninf=noninf(d))
    ph, pmh = holm([out[c]['p'] for c in conds]), holm([out[c]['p_map'] for c in conds])
    for k, c in enumerate(conds):
        out[c]['p_holm'], out[c]['p_map_holm'] = float(ph[k]), float(pmh[k])
    return out


res = dict(frozen=FZ, sws_recalc_max_err=float(err))
res['means'] = {c: {m: dict(mean=float(vec(c, m).mean()), ci=boot(vec(c, m)),
                            found=float(np.mean([data[c][m][i]['found_frac'] for i in insts(c)])))
                    for m in data[c]} for c in data}
res['avg7'] = {m: float(np.mean([vec(c, m).mean() for c in CONDS])) for m in data['R00'] if m in data['S1']}
PAIRS = [('HBF', 'HEDAC*-eps'), ('HBF', 'HEDAC*'), ('HBF', 'BSS'), ('HBF', 'HBF-fix'), ('HEDAC*-eps', 'HEDAC*'),
         ('HEDAC*-eps', 'BSS')]
res['primary'] = {f'{a}|{b}': primary(a, b) for a, b in PAIRS}
fam = ['HBF|HEDAC*-eps', 'HBF|HEDAC*', 'HBF|BSS', 'HBF|HBF-fix']
for k, v in zip(fam, holm([res['primary'][k]['p_perm'] for k in fam])):
    res['primary'][k]['p_holm'] = float(v)
res['cond'] = {f'{a}|{b}': cond_cmp(a, b, CONDS + (['S3c'] if b in data['S3c'] and a in data['S3c'] else []))
               for a, b in PAIRS}
# H4: не-гіршість HBF відносно HEDAC* за правильної карти
res['H4'] = {c: res['cond']['HBF|HEDAC*'][c]['p_noninf'] for c in ('S1', 'S2', 'R00')}
# відставання R від найкращого з оцінених статичних ε у кожній умові
eps_labels = sorted({m for c in CONDS for m in data[c] if m.startswith('HEDAC*|e')})
pool = {c: [m for m in data[c] if m.startswith('HEDAC*') and (m in eps_labels or m in ('HEDAC*', 'HEDAC*-eps'))] for c in CONDS}
reg = {}
for m in res['avg7']:
    rr = {c: float(max(vec(c, k).mean() for k in pool[c]) - vec(c, m).mean()) for c in CONDS}
    reg[m] = dict(per=rr, max=max(rr.values()))
res['regret'] = reg
res['regret_pool'] = pool
# H6: чутливість — сімейства апріорних налаштувань HBF і значень ε
fam_hbf = ['HBF'] + sens_names
fam_eps = [f'HEDAC*|e{etag(e)}' for e in FZ['sens_eps']]
J = {m: float(np.mean([vec(c, m).mean() for c in CONDS])) for m in fam_hbf + fam_eps}
res['H6'] = dict(J=J, spread_hbf=max(J[m] for m in fam_hbf) - min(J[m] for m in fam_hbf),
                 spread_eps=max(J[m] for m in fam_eps) - min(J[m] for m in fam_eps),
                 worst_hbf=min(J[m] for m in fam_hbf), hedac_default=J[f'HEDAC*|e{etag(0.05)}'],
                 best_eps=max(fam_eps, key=lambda m: J[m]))
res['H6']['worst_hbf_vs_best_eps'] = primary(min(fam_hbf, key=lambda m: J[m]), res['H6']['best_eps'], nperm=20000, nboot=4000)
# чутливість до τ
res['tau'] = {str(int(t)): {f'{a}|{b}': primary(a, b, tau=t, nperm=20000, nboot=4000) for a, b in PAIRS[:4]} for t in TAUS}
res['tau_means'] = {str(int(t)): {m: float(np.mean([vec(c, m, t).mean() for c in CONDS])) for m in MAIN} for t in TAUS}
# обчислювальна вартість — лише прогони, виконані на одній машині (нові)
res['ms_per_step'] = {m: float(np.median([data[c][m][i]['ms_per_step'] for c in CONDS for i in insts(c)]))
                      for m in ('HBF', 'HBF-fix', 'HEDAC*-eps')}
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok', OUT, 'max SWS recompute err', err)
for k in fam:
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in res['primary'][k].items()})
print('avg7', {k: round(v, 4) for k, v in res['avg7'].items()})
print('H4', res['H4'])
print('H6', {k: v for k, v in res['H6'].items() if k != 'J'}, {k: round(v, 4) for k, v in J.items()})
