# -*- coding: utf-8 -*-
"""
Масштабування за розміром області (Е9): сітки 50x50 (контроль), 100x100, 200x200 — БЕЗ зміни env.py.

Як це працює. Файли моделей мають сталі модуля (W = H = 50, N, RC, ...), які методи імпортують під час
завантаження. Тому до імпорту будь-якого модуля моделей процес викликає install(size): вихідний текст
env.py (з диска, без змін) виконується в пам'яті з рядком «W = H = <size>», і отриманий модуль реєструється
як 'env'. Файл env.py на диску не змінюється, тож хеш коду наявних експериментів той самий; хеш
експериментів Е9 додатково містить цей файл.

Масштабована реалізація (scalable=True) замість щільних матриць N x N:
  * apsp  -> LazyAPSP: відстані BFS (8-зв'язність) від однієї клітинки на вимогу, з LRU-кешем; ЦІЛІ числа —
             побітово ті самі, що в щільній матриці;
  * G     -> GreenOp: той самий оператор (λI + D − A₄)⁻¹ з маскою вільних клітинок, але через розріджену
             LU-факторизацію (scipy.sparse.linalg.splu) — так HEDAC розв'язує рівняння в оригінальних роботах.
             Відмінність від щільного оберненого — лише похибка округлення (≈1e-7 відносно).
  Час на обчислення відстаней і розв'язання рівняння рахується окремо (bfs_ms_per_step, solve_ms_per_step),
  він входить у ms_per_step відповідних методів (це їхня реальна вартість на великих картах).
Щільна реалізація (scalable=False) — вихідна, лише для порівняння (≈1 ГБ ОЗП на карту 100x100).

Масштабування задачі з коефіцієнтом s = size / 50 («та сама задача, більша область»): розміри й відстані
генератора карт (заборонені зони, апріор, текстура рельєфу) множаться на s (кількість зон та сама, тож
частка забороненої площі зберігається); сенсор (слід 3x3, POD) і рух агента (1 клітинка за крок) не змінюються. Бюджет кроків T і параметр
τ кривої виживання задаються в конфігурації (у програмі Е9: T = 300·s, τ = 180·s). За s = 1 генератор
побітово відтворює вихідні карти (перевіряє тест).
"""
import os
import sys
import time
import types
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_STATE = {}


def install(size=50, scalable=True):
    """Підміняє модуль env версією з сіткою size x size. Викликати ДО імпорту моделей."""
    size = int(size)
    if 'env' in sys.modules:
        cur = sys.modules['env']
        if getattr(cur, '_GRID_INSTALLED', None) == (size, bool(scalable)):
            return cur
        if getattr(cur, 'W', None) == size and not scalable and not hasattr(cur, '_GRID_INSTALLED'):
            return cur                                    # звичайний env.py 50x50, щільна реалізація
        raise RuntimeError(f'модуль env уже завантажено (W={getattr(cur, "W", "?")}); gridscale.install({size}) '
                           f'треба викликати на початку процесу')
    path = os.path.join(HERE, 'env.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    key = 'W = H = 50\n'
    if src.count(key) != 1:
        raise RuntimeError('у env.py не знайдено рядка «W = H = 50» — gridscale несумісний з цією версією')
    src = src.replace(key, f'W = H = {size}\n', 1)
    mod = types.ModuleType('env')
    mod.__file__ = path
    sys.modules['env'] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)
    s = size / 50.0
    base_cls = mod.MapInstance
    mod.MapInstanceDense = base_cls
    mod.MapInstance = _make_scaled_class(mod, base_cls, s, scalable)
    mod._GRID_INSTALLED = (size, bool(scalable))
    mod.GRID_SCALE = s
    _STATE.update(size=size, scalable=bool(scalable))
    return mod


def _si(x, s):
    return int(round(x * s))


def _make_scaled_class(E, Base, s, scalable):
    H, W, N, RR, CC = E.H, E.W, E.N, E.RR, E.CC

    class ScaledMapInstance(Base):
        """Карта з масштабованим генератором; за s = 1 — побітово як вихідна."""
        GRID_S = s
        SCALABLE = scalable

        def _gen_terrain(self, rng):
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(rng.standard_normal((H, W)), 4 * s)
            forest_share = {'S1': 0.10, 'S2': 0.30, 'S3': 0.35}[self.scenario]
            shrub_share = {'S1': 0.15, 'S2': 0.20, 'S3': 0.20}[self.scenario]
            q_f = np.quantile(noise, 1 - forest_share)
            q_s = np.quantile(noise, 1 - forest_share - shrub_share)
            terr = np.zeros((H, W), int)
            terr[noise > q_s] = 1
            terr[noise > q_f] = 2
            self.terrain = terr.ravel()

        def _gen_obstacles(self, rng):
            from scipy.ndimage import label
            free = np.ones((H, W), bool)
            n_zones = {'S1': 1, 'S2': 3, 'S3': 6}[self.scenario]      # розміри зон масштабуються -> частка площі та сама
            for _ in range(n_zones):
                if rng.random() < 0.5:
                    r0, c0 = rng.integers(_si(3, s), H - _si(12, s)), rng.integers(_si(3, s), W - _si(10, s))
                    h, w = rng.integers(_si(3, s), _si(9, s)), rng.integers(_si(3, s), _si(12, s))
                    free[r0:r0 + h, c0:c0 + w] = False
                else:
                    r0, c0, rad = (rng.integers(_si(5, s), H - _si(8, s)), rng.integers(_si(5, s), W - _si(5, s)),
                                   rng.integers(_si(2, s), _si(5, s)))
                    free[(RR.reshape(H, W) - r0) ** 2 + (CC.reshape(H, W) - c0) ** 2 <= rad * rad] = False
            base_r, base_c = H - 2, int(rng.integers(W // 4, 3 * W // 4))
            free[base_r - 1:base_r + 2, base_c - 1:base_c + 2] = True
            lab, _ = label(free, structure=np.ones((3, 3)))
            free = lab == lab[base_r, base_c]
            self.free = free.ravel()
            self.base = E.idx(base_r, base_c)

        def _gen_prior(self, rng):
            g = np.zeros((H, W))
            R2, C2 = RR.reshape(H, W), CC.reshape(H, W)

            def gauss(r, c, sg, w):
                return w * np.exp(-((R2 - r) ** 2 + (C2 - c) ** 2) / (2 * sg * sg))

            if self.scenario == 'S1':
                r, c = rng.integers(_si(8, s), _si(28, s)), rng.integers(_si(10, s), _si(40, s))
                g += gauss(r, c, 6.0 * s, 1.0)
            else:
                r, c = rng.integers(_si(8, s), _si(30, s)), rng.integers(_si(8, s), _si(42, s))
                g += gauss(r, c, 4.0 * s, 1.0) / (2 * np.pi * 16 * s * s)
                r1, c1, r2, c2 = (rng.integers(_si(3, s), _si(35, s)), rng.integers(0, _si(10, s)),
                                  rng.integers(_si(3, s), _si(35, s)), rng.integers(_si(40, s), _si(50, s)))
                tt = np.linspace(0, 1, _si(60, s))
                line = np.zeros((H, W))
                for a in tt:
                    line += gauss(r1 + a * (r2 - r1), c1 + a * (c2 - c1), 1.8 * s, 1.0)
                g += 1.3 * line / line.sum()
                rr0, amp, ph = rng.integers(_si(10, s), _si(35, s)), rng.uniform(2 * s, 6 * s), rng.uniform(0, 6.28)
                river_r = rr0 + amp * np.sin(np.linspace(0, 2 * np.pi, W) + ph)
                band = np.exp(-(R2 - river_r[None, :]) ** 2 / (2 * (1.5 * s) ** 2))
                g += 1.0 * band / band.sum()
                r, c = rng.integers(_si(5, s), _si(38, s)), rng.integers(_si(5, s), _si(45, s))
                sb = gauss(r, c, 2.2 * s, 1.0)
                g += 0.8 * sb / sb.sum()
            g = g.ravel() * self.free
            self.prior = g / g.sum()

        def _precompute(self, lam_len):
            if not scalable:
                return Base._precompute(self, lam_len)
            precompute_scalable(E, self, lam_len)

        # --- облік часу на відстані / рівняння в межах одного прогону (для ms_per_step) ---
        def begin_run(self, clear=True):
            if hasattr(self.apsp, 'begin_run'):
                self.apsp.begin_run(clear)
            if hasattr(self.G, 'begin_run'):
                self.G.begin_run()

        def run_counters(self, T):
            out = {}
            if hasattr(self.apsp, 't_run'):
                out.update(bfs_ms_per_step=1000 * self.apsp.t_run / T, bfs_calls=int(self.apsp.n_run))
            if hasattr(self.G, 't_run'):
                out.update(solve_ms_per_step=1000 * self.G.t_run / T, solve_calls=int(self.G.n_run))
            return out

    return ScaledMapInstance


def precompute_scalable(E, mp, lam_len):
    """Та сама передобробка, що MapInstance._precompute, але без щільних матриць N x N."""
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix, diags
    H, W, N, RR, CC = E.H, E.W, E.N, E.RR, E.CC
    nbr = np.full((N, 8), -1, int)
    for k, (dr, dc) in enumerate(E.MOVES):
        r2, c2 = RR + dr, CC + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.where(ok, r2 * W + c2, 0)
        ok &= mp.free[j] & mp.free
        nbr[ok, k] = j[ok]
    mp.nbr = nbr
    F = len(E.FP_OFF)
    fp_idx = np.full((N, F), N, int)
    fp_p = np.zeros((N, F))
    for k, (dr, dc) in enumerate(E.FP_OFF):
        r2, c2 = RR + dr, CC + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.where(ok, r2 * W + c2, 0)
        ok &= mp.free[j]
        fp_idx[ok, k] = j[ok]
        fp_p[ok, k] = E.P0 * E.TERRAIN_FACTOR[mp.terrain[j[ok]]] * E.FP_W[k]
    mp.fp_idx, mp.fp_p = fp_idx, fp_p
    mp.pdet = E.P0 * E.TERRAIN_FACTOR[mp.terrain] * mp.free
    offs = np.array([(dr, dc) for dr in range(-E.WIN_R, E.WIN_R + 1) for dc in range(-E.WIN_R, E.WIN_R + 1)])
    win = np.full((N, len(offs)), N, np.int32 if N < 2 ** 31 else int)
    for k, (dr, dc) in enumerate(offs):
        r2, c2 = RR + dr, CC + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.where(ok, r2 * W + c2, 0)
        ok &= mp.free[j]
        win[ok, k] = j[ok]
    mp.win = win.astype(int) if N <= 2500 else win
    rows, cols = np.nonzero(nbr >= 0)
    A = coo_matrix((np.ones(len(rows)), (rows, nbr[rows, cols])), shape=(N, N)).tocsr()
    mp.apsp = LazyAPSP(A, N)
    _, inds = distance_transform_edt(~mp.free.reshape(H, W), return_indices=True)
    mp.nearest_free = (inds[0] * W + inds[1]).ravel()
    lam = 1.0 / lam_len ** 2
    r4, c4 = [], []
    deg = np.zeros(N)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r2, c2 = RR + dr, CC + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.where(ok, r2 * W + c2, 0)
        ok &= mp.free[j] & mp.free
        r4.append(np.nonzero(ok)[0])
        c4.append(j[ok])
        deg[ok] += 1
    r4, c4 = np.concatenate(r4), np.concatenate(c4)
    Mx = (diags(lam + deg) - coo_matrix((np.ones(len(r4)), (r4, c4)), shape=(N, N))).tocsc()
    mp.G = GreenOp(Mx, mp.free)


class LazyAPSP:
    """Матриця найкоротших відстаней на вимогу: рядок = BFS від клітинки; цілі числа як у щільній apsp."""

    def __init__(self, A, N, cache_mb=64):
        self.A, self.N = A, N
        self.cap = max(64, int(cache_mb * 1e6 / (2 * N)))
        self.cache = OrderedDict()
        self.t_run = 0.0
        self.n_run = 0

    def begin_run(self, clear=True):
        if clear:
            self.cache.clear()                 # кожен прогін рахує власні відстані (чесний облік часу)
        self.t_run, self.n_run = 0.0, 0

    def row(self, c):
        c = int(c)
        r = self.cache.get(c)
        if r is not None:
            self.cache.move_to_end(c)
            return r
        t0 = time.perf_counter()
        from scipy.sparse.csgraph import dijkstra
        d = dijkstra(self.A, directed=False, unweighted=True, indices=c)
        d[~np.isfinite(d)] = 9999
        r = d.astype(np.int16)
        self.t_run += time.perf_counter() - t0
        self.n_run += 1
        self.cache[c] = r
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return r

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.row(key)
        if isinstance(key, tuple) and len(key) == 2:
            a, b = key
            if isinstance(a, np.ndarray) and a.ndim == 2:          # np.ix_(рядки, стовпці)
                ra, cb = a.ravel(), np.asarray(b).ravel()
                if len(set(ra.tolist())) <= len(set(cb.tolist())):
                    return np.stack([self.row(x)[cb] for x in ra])
                return np.stack([self.row(x)[ra] for x in cb]).T   # відстані симетричні
            if np.ndim(a) == 0:
                return self.row(a)[b]
            if np.ndim(b) == 0:
                return self.row(b)[a]
        raise TypeError(f'LazyAPSP: непідтримуваний індекс {type(key)}')


class GreenOp:
    """Оператор G = F (λI + D − A₄)⁻¹ F (F — маска вільних клітинок) через розріджену LU-факторизацію."""

    def __init__(self, Mx, free):
        from scipy.sparse.linalg import splu
        self.lu = splu(Mx)
        self.f = free.astype(np.float64)
        self.N = len(free)
        self.t_run = 0.0
        self.n_run = 0

    def begin_run(self):
        self.t_run, self.n_run = 0.0, 0

    def _solve(self, B):
        t0 = time.perf_counter()
        X = self.lu.solve(B)
        self.t_run += time.perf_counter() - t0
        self.n_run += 1
        return X

    def __matmul__(self, v):
        x = self._solve(np.asarray(v, np.float64) * self.f)
        return (x * self.f).astype(np.float32)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            e = np.zeros(self.N)
            e[int(key)] = self.f[int(key)]
            return (self._solve(e) * self.f).astype(np.float32)
        idx = np.asarray(key, int).ravel()
        B = np.zeros((self.N, len(idx)))
        B[idx, np.arange(len(idx))] = self.f[idx]
        X = self._solve(B)
        return (X * self.f[:, None]).T.astype(np.float32)
