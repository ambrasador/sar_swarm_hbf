# -*- coding: utf-8 -*-
"""
Кейс на реальній місцевості (Е11): карта з даних OpenStreetMap (realmaps/<name>/osm.json, див. realmap_fetch.py).

  python realmap.py --name karpaty --size 100      — растеризація, статистика шарів, рисунок карти
                                                     (realmaps/karpaty/map_100.png)

Растеризація (лише numpy / scipy / matplotlib) у квадратну сітку size x size (рядок 0 — північ):
  рельєф для POD (як у синтетичній моделі): 2 — ліс (natural=wood, landuse=forest); 1 — чагарник / болото /
  сади (natural=scrub|heath|wetland, landuse=orchard|vineyard); 0 — відкрита місцевість;
  вода (natural=water, landuse=reservoir|basin, waterway=riverbank) — польоти дозволені, апріор = 0;
  заборонені для польотів зони: аеродроми (aeroway=aerodrome); ЛЕП (power=line) — лише інформаційний шар
  (БпЛА перетинають їх на безпечній висоті, тож вони не розрізають область);
  лінійні орієнтири: дороги, стежки (highway=track|path|footway|bridleway|cycleway|steps), струмки / річки,
  будівлі.
Сценарії (номер «карти» = номер інциденту на тій самій місцевості, різні LKP):
  S1 — одна LKP (на стежці / дорозі в центральній частині), апріор — гаусіан σ = 6·s клітинок;
  S2 — LKP (σ = 4·s) + стежки/дороги (1.3) + струмки (1.0) + будівлі (0.8) у радіусі 20·s клітинок від LKP
       (ті самі ваги, що в синтетичному S2, але з РЕАЛЬНИХ об'єктів);
  S3 — апріор S2, але 40% постраждалих поза моделлю (як у синтетичному S3).
Масштаб s = size / 50; рух і сенсор — як у моделі. Передобчислення — масштабоване (gridscale).
Дані: © OpenStreetMap contributors (ODbL).
"""
import argparse, hashlib, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LAYERS = {}
FOREST = {('natural', 'wood'), ('landuse', 'forest')}
SHRUB = {('natural', 'scrub'), ('natural', 'heath'), ('natural', 'wetland'), ('landuse', 'orchard'),
         ('landuse', 'vineyard')}
WATER = {('natural', 'water'), ('landuse', 'reservoir'), ('landuse', 'basin'), ('waterway', 'riverbank')}
PATHS = {'track', 'path', 'footway', 'bridleway', 'cycleway', 'steps'}
STREAMS = {'river', 'stream', 'canal', 'ditch', 'drain'}


def map_dir(name):
    return os.path.join(HERE, 'realmaps', name)


def load_meta(name):
    with open(os.path.join(map_dir(name), 'meta.json'), encoding='utf-8') as f:
        return json.load(f)


def osm_sha(name):
    with open(os.path.join(map_dir(name), 'osm.json'), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(real):
    """Перевірка наявності й цілісності даних карти. -> повідомлення про помилку або None."""
    name = real.get('name')
    p = os.path.join(map_dir(name), 'osm.json')
    if not os.path.exists(p) or not os.path.exists(os.path.join(map_dir(name), 'meta.json')):
        return f'немає даних карти realmaps/{name}/ — виконайте realmap_fetch.py (див. ДОДАТКОВІ_ЕКСПЕРИМЕНТИ.md)'
    if real.get('sha256') and osm_sha(name) != real['sha256']:
        return f'дані карти realmaps/{name}/osm.json змінено (sha256 не збігається з конфігом)'
    return None


# ------------------------------------------------------------------ геометрія
def _projector(meta, size):
    s, w, n, e = meta['bbox']
    lat0 = (s + n) / 2
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    side = max((e - w) * kx, (n - s) * ky)
    cell = side / size

    def to_xy(lat, lon):                                  # x — стовпець, y — рядок (0 — північ), у клітинках
        return (np.asarray(lon) - w) * kx / cell, (n - np.asarray(lat)) * ky / cell
    return to_xy, cell


def _rings(member_ways):
    """Збирає замкнені кільця з відрізків (списків вузлів) за збігом кінців."""
    segs = [list(s) for s in member_ways if len(s) >= 2]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for k, sg in enumerate(segs):
                if sg[0] == ring[-1]:
                    ring += sg[1:]
                elif sg[-1] == ring[-1]:
                    ring += sg[::-1][1:]
                elif sg[-1] == ring[0]:
                    ring = sg[:-1] + ring
                elif sg[0] == ring[0]:
                    ring = sg[::-1][:-1] + ring
                else:
                    continue
                segs.pop(k)
                changed = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def _classify(tags):
    t = {(k, v) for k, v in tags.items()}
    if t & FOREST:
        return 'area', 'forest'
    if t & WATER:
        return 'area', 'water'
    if t & SHRUB:
        return 'area', 'shrub'
    if 'building' in tags:
        return 'area', 'building'
    if tags.get('aeroway') == 'aerodrome':
        return 'area', 'nofly'
    if 'highway' in tags:
        return 'line', 'path' if tags['highway'] in PATHS else 'road'
    if tags.get('waterway') in STREAMS:
        return 'line', 'stream'
    if tags.get('power') == 'line':
        return 'line', 'power'
    return None, None


def rasterize(name, size):
    """-> dict шарів (size x size) + cell_m. Кешується в межах процесу."""
    key = (name, int(size))
    if key in _LAYERS:
        return _LAYERS[key]
    from matplotlib.path import Path
    meta = load_meta(name)
    with open(os.path.join(map_dir(name), 'osm.json'), encoding='utf-8') as f:
        els = json.load(f)['elements']
    nodes = {e['id']: (e['lat'], e['lon']) for e in els if e['type'] == 'node' and 'lat' in e}
    ways = {e['id']: e for e in els if e['type'] == 'way'}
    to_xy, cell = _projector(meta, size)
    L = {k: np.zeros((size, size), bool) for k in ('forest', 'shrub', 'water', 'building', 'nofly', 'road', 'path',
                                                    'stream', 'power')}
    yy, xx = np.mgrid[0:size, 0:size]
    pts_all = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)

    def xy_of(nids):
        ll = [nodes[i] for i in nids if i in nodes]
        if len(ll) < 2:
            return None
        lat, lon = np.array(ll).T
        x, y = to_xy(lat, lon)
        return np.stack([x, y], 1)

    def fill(poly_xy, out, value=True):
        if poly_xy is None or len(poly_xy) < 3:
            return
        x0, y0 = np.floor(poly_xy.min(0)).astype(int)
        x1, y1 = np.ceil(poly_xy.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size), min(y1, size)
        if x0 >= x1 or y0 >= y1:
            return
        sub = (yy[y0:y1, x0:x1].ravel(), xx[y0:y1, x0:x1].ravel())
        inside = Path(poly_xy).contains_points(np.stack([sub[1] + 0.5, sub[0] + 0.5], 1))
        out[sub[0][inside], sub[1][inside]] = value

    def draw(line_xy, out):
        if line_xy is None:
            return
        for (xa, ya), (xb, yb) in zip(line_xy[:-1], line_xy[1:]):
            n = max(2, int(math.hypot(xb - xa, yb - ya) * 4) + 1)
            xs, ys = np.linspace(xa, xb, n), np.linspace(ya, yb, n)
            ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
            out[ys[ok].astype(int), xs[ok].astype(int)] = True

    for w in ways.values():
        kind, cls = _classify(w.get('tags', {}))
        if kind is None:
            continue
        xy = xy_of(w['nodes'])
        if kind == 'area' and w['nodes'][0] == w['nodes'][-1]:
            fill(xy, L[cls])
        elif kind == 'line':
            draw(xy, L[cls])
    for r in (e for e in els if e['type'] == 'relation'):
        kind, cls = _classify(r.get('tags', {}))
        if kind != 'area':
            continue
        outer = [ways[m['ref']]['nodes'] for m in r.get('members', [])
                 if m['type'] == 'way' and m.get('role') in ('outer', '') and m['ref'] in ways]
        inner = [ways[m['ref']]['nodes'] for m in r.get('members', [])
                 if m['type'] == 'way' and m.get('role') == 'inner' and m['ref'] in ways]
        tmp = np.zeros((size, size), bool)
        for ring in _rings(outer):
            fill(xy_of(ring), tmp)
        for ring in _rings(inner):
            fill(xy_of(ring), tmp, False)
        L[cls] |= tmp
    terrain = np.zeros((size, size), int)
    terrain[L['shrub']] = 1
    terrain[L['forest']] = 2
    terrain[L['water']] = 0
    nofly = L['nofly'].copy()          # ЛЕП — лише інформаційний шар: БпЛА перетинають їх на безпечній висоті
    out = dict(terrain=terrain, water=L['water'], nofly=nofly, road=L['road'], path=L['path'], stream=L['stream'],
               building=L['building'], power=L['power'], cell_m=cell, size=int(size), meta=meta)
    _LAYERS[key] = out
    return out


# ------------------------------------------------------------------ карта
def make_instance(real, scenario, seed, lam_len=5.0):
    """Екземпляр карти на реальній місцевості (модуль env має бути встановлено gridscale з потрібним size)."""
    E = sys.modules.get('env')
    if E is None or not hasattr(E, '_GRID_INSTALLED'):
        raise RuntimeError('реальна карта потребує gridscale.install(size) до імпорту моделей')
    lay = rasterize(real['name'], E.W)
    cls = _real_class(E)
    return cls(scenario, seed, lam_len=lam_len, layers=lay, real=real)


_CLS = {}


def _real_class(E):
    if E.W in _CLS:
        return _CLS[E.W]
    from scipy.ndimage import gaussian_filter, label
    H, W, N, RR, CC = E.H, E.W, E.N, E.RR, E.CC
    s = W / 50.0

    class RealMapInstance(E.MapInstance):
        def __init__(self, scenario, seed, lam_len=5.0, layers=None, real=None):
            self.layers, self.real = layers, real or {}
            super().__init__(scenario, seed, lam_len)

        def _gen_terrain(self, rng):
            self.terrain = self.layers['terrain'].ravel().astype(int)

        def _gen_obstacles(self, rng):
            lay = self.layers
            free = ~lay['nofly']
            meta = lay['meta']
            if meta.get('base'):
                to_xy, _ = _projector(meta, W)
                x, y = to_xy(*meta['base'])
                br, bc = int(np.clip(y, 1, H - 2)), int(np.clip(x, 1, W - 2))
            else:                                               # найближча дорога / стежка до середини південного краю
                lin = np.nonzero((lay['road'] | lay['path']).ravel() & free.ravel())[0]
                if len(lin):
                    d = (RR[lin] - (H - 2)) ** 2 + (CC[lin] - W / 2) ** 2
                    b = lin[np.argmin(d)]
                    br, bc = int(np.clip(RR[b], 1, H - 2)), int(np.clip(CC[b], 1, W - 2))
                else:
                    br, bc = H - 2, W // 2
            free[br - 1:br + 2, bc - 1:bc + 2] = True
            lab, _ = label(free, structure=np.ones((3, 3)))
            free = lab == lab[br, bc]
            self.free = free.ravel()
            self.base = E.idx(br, bc)

        def _gen_prior(self, rng):
            lay = self.layers
            land = self.free & ~lay['water'].ravel()
            R2, C2 = RR.reshape(H, W), CC.reshape(H, W)
            central = (RR >= 0.15 * H) & (RR <= 0.85 * H) & (CC >= 0.15 * W) & (CC <= 0.85 * W)
            far = np.hypot(RR - RR[self.base], CC - CC[self.base]) >= 0.2 * W
            lin = (lay['road'] | lay['path']).ravel()
            cand = np.nonzero(land & central & far & lin)[0]
            if len(cand) < 20:
                cand = np.nonzero(land & central)[0]
            lkp = int(cand[rng.integers(len(cand))])
            r, c = RR[lkp], CC[lkp]
            self.lkp = lkp

            def gauss(sg):
                return np.exp(-((R2 - r) ** 2 + (C2 - c) ** 2) / (2 * sg * sg))

            if self.scenario == 'S1':
                g = gauss(6.0 * s)
            else:
                g = gauss(4.0 * s)
                g = g / g.sum()
                near = (np.hypot(R2 - r, C2 - c) <= 20 * s)
                for layer, sg, wgt in (('path', 1.8, 1.3), ('road', 1.8, 0.0), ('stream', 1.5, 1.0),
                                       ('building', 2.2, 0.8)):
                    if layer == 'path':
                        m = (lay['path'] | lay['road']) & near
                    elif layer == 'road':
                        continue
                    else:
                        m = lay[layer] & near
                    if m.any() and wgt > 0:
                        b = gaussian_filter(m.astype(float), sg * s)
                        g = g + wgt * b / b.sum()
            g = g.ravel() * land
            self.prior = g / g.sum()

    _CLS[E.W] = RealMapInstance
    return RealMapInstance


# ------------------------------------------------------------------ перевірка / рисунок
def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--size', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import gridscale
    gridscale.install(a.size)
    import env as E
    real = dict(name=a.name)
    err = check(real)
    if err:
        print('ПОМИЛКА:', err)
        return 1
    lay = rasterize(a.name, a.size)
    meta = lay['meta']
    n = a.size * a.size
    print(f"Карта {a.name}: {a.size}x{a.size} клітинок по {lay['cell_m']:.0f} м "
          f"({a.size * lay['cell_m'] / 1000:.1f} x {a.size * lay['cell_m'] / 1000:.1f} км), sha256 {osm_sha(a.name)[:12]}…")
    t = lay['terrain']
    print(f"  ліс {100 * (t == 2).mean():.1f}%, чагарник/болото {100 * (t == 1).mean():.1f}%, вода {100 * lay['water'].mean():.1f}%, "
          f"заборонено (аеродроми) {100 * lay['nofly'].mean():.1f}%, ЛЕП {lay['power'].sum()} клітинок")
    print(f"  клітинок: доріг {lay['road'].sum()}, стежок {lay['path'].sum()}, струмків {lay['stream'].sum()}, "
          f"будівель {lay['building'].sum()}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    for k, sc in enumerate(['S1', 'S2']):
        mp = make_instance(real, sc, a.seed)
        ax[k + 1].imshow(mp.prior.reshape(E.H, E.W), cmap='magma')
        ax[k + 1].plot(E.CC[mp.lkp], E.RR[mp.lkp], 'c*', ms=14, label='LKP')
        ax[k + 1].plot(E.CC[mp.base], E.RR[mp.base], 'ws', ms=9, label='база')
        ax[k + 1].set_title(f'Апріор {sc} (інцидент {a.seed})')
        ax[k + 1].legend(loc='lower right', fontsize=8)
        if k == 0:
            print(f"  вільних клітинок {mp.free.sum()} з {n}; база ({E.RR[mp.base]}, {E.CC[mp.base]})")
    base_img = t.astype(float)
    ax[0].imshow(base_img, cmap=ListedColormap(['#e8f0d8', '#b5cf8a', '#3f7a3a']), vmin=0, vmax=2)
    for layer, col in (('water', '#4a90d9'), ('stream', '#1f5fa8'), ('path', '#8b5a2b'), ('road', '#444444'),
                       ('building', '#d62728'), ('power', '#ff00ff'), ('nofly', '#000000')):
        yy, xx = np.nonzero(lay[layer])
        ax[0].scatter(xx, yy, s=1.2 if layer != 'water' else 2, c=col, marker='s', linewidths=0, label=layer)
    ax[0].set_title(f'{a.name}: ліс / чагарник / відкрито + об\'єкти OSM')
    ax[0].legend(loc='lower right', fontsize=7, markerscale=5)
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.text(0.01, 0.01, meta.get('attribution', '© OpenStreetMap contributors'), fontsize=8)
    plt.tight_layout()
    out = os.path.join(map_dir(a.name), f'map_{a.size}.png')
    plt.savefig(out, dpi=160)
    print('Рисунок:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
