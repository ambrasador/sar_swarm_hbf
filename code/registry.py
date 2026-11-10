# -*- coding: utf-8 -*-
"""
Реєстр методів і груп (використовується раннером та аналізом).

ORDER / METHODS / BY_NAME — рівно 14 методів експерименту main (як і раніше: "all" = ці 14).
Розширення v2 (варіанти для абляцій, суфікси):
  <назва>-C — децентралізований варіант (контролер-репліка, обмежений зв'язок, decentral.py);
  <назва>-U — метод без обізнаності про відмови (сирий метод у середовищі з відмовами);
  <назва>   у середовищі зв'язку/відмов — оракульський варіант (миттєво злиті знання).
"""
import baselines as B, novel as NV, hybrid as HB

EXISTING = [B.Lawnmower, B.Levy, B.PSO, B.GWO, B.ACO, B.Greedy, B.HEDAC]
NEW = [NV.SIFD, NV.PAS, NV.BSS]
MODIF = [NV.PPS]
HYBRID = [HB.GWO_PF, HB.ACO_IFD, HB.PSO_LS]
METHODS = EXISTING + NEW + MODIF + HYBRID
BY_NAME = {m.name: m for m in METHODS}
ORDER = [m.name for m in METHODS]
_BASE_GROUP = {m.name: g for g, ms in [('існуючий', EXISTING), ('новий', NEW), ('модифікація', MODIF),
                                       ('гібридний', HYBRID)] for m in ms}


def _variants():
    import variants
    return variants


def group_of(name):
    if name in _BASE_GROUP:
        return _BASE_GROUP[name]
    try:
        _, _, suf = resolve(name)
    except KeyError:
        return 'інший'
    if suf == '-C':
        return 'децентралізований'
    if suf == '-U':
        return 'без обізнаності'
    return 'варіант'


class _Groups(dict):
    """dict груп із запасним визначенням для нових назв (сумісно з GROUP[m] і pandas .map)."""
    def __missing__(self, key):
        return group_of(key)


GROUP = _Groups(_BASE_GROUP)


def base_name(name):
    """Назва базового методу: 'SIFD-noS-C' -> 'SIFD', 'GREEDY-U' -> 'GREEDY'."""
    return _variants().base_of(name)


def base_group(name):
    """Група базового методу (для аналізу децентралізованих варіантів і варіантів абляцій)."""
    return _BASE_GROUP.get(base_name(name), 'інший')


def expand_methods(spec):
    """'all' | 'all-C' | 'all-U' | список (може містити 'all', 'all-C', 'all-U')."""
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for s in spec:
        if s == 'all':
            out += ORDER
        elif s == 'all-C':
            out += [m + '-C' for m in ORDER]
        elif s == 'all-U':
            out += [m + '-U' for m in ORDER]
        else:
            out.append(s)
    seen, res = set(), []
    for m in out:
        if m not in seen:
            seen.add(m); res.append(m)
    return res


def resolve(name):
    """-> (клас, параметри, суфікс '' | '-C' | '-U')."""
    suf = ''
    base = name
    V0 = _variants().VARIANTS
    if name not in BY_NAME and name not in V0:
        for s in ('-C', '-U'):
            if name.endswith(s):
                suf, base = s, name[:-len(s)]
    if base in BY_NAME:
        return BY_NAME[base], {}, suf
    V = _variants().VARIANTS
    if base in V:
        return V[base][0], dict(V[base][1]), suf
    raise KeyError(f'Невідомий метод: {name}')


def order_names(names):
    """Порядок для таблиць: 14 базових, варіанти, потім -C / -U, решта — за алфавітом."""
    V = list(_variants().VARIANTS)
    base_rank = {m: i for i, m in enumerate(ORDER + V)}

    def key(n):
        b, s = n, 0
        if n.endswith('-C'):
            b, s = n[:-2], 1
        elif n.endswith('-U'):
            b, s = n[:-2], 2
        return (s, base_rank.get(b, 10 ** 6), n)
    return sorted(set(names), key=key)
