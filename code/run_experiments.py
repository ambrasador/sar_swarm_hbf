# -*- coding: utf-8 -*-
"""
Відмовостійкий раннер експериментів v2 (відновлення після вимкнення ПК / збою живлення).

Принципи (як у v1):
  * кожен завершений прогін (сценарій, карта, розстановка, метод) одразу дописується
    в results/<exp>/runs.jsonl з flush + fsync — втрачається щонайбільше поточне завдання;
  * під час запуску журнал читається, пошкоджений «хвіст» обрізається, виконані прогони
    пропускаються — робота продовжується з місця зупинки;
  * результати детерміновані (зерна залежать лише від ключа прогону);
  * файл блокування з «серцебиттям» (окремий потік) не дає запустити два раннери на один експеримент;
  * хеш вихідного коду фіксується в config.json; якщо код змінився — продовження блокується;
  * м'яка зупинка: Ctrl+C або файл STOP у каталозі експерименту;
  * progress.json оновлюється атомарно (tmp + os.replace з повторами для Windows).

Нове у v2:
  * хеш коду = 4 файли моделей + лише ті модулі розширень, які реально використовує експеримент
    (ext_env.py, variants.py, decentral.py, probes.py). Для «базових» експериментів (main, scal_*)
    хеш збігається з v1 (0fb359d0562b268a), тож результати v1 і v2 можна звірити побітово;
    окремо записується code_hash_base (лише 4 файли моделей);
  * фабрика середовища: base (env.SAREnv), ext (row / rho_out / survival), dyn (рухомі постраждалі),
    comm (обмежений зв'язок і відмови агентів);
  * фабрика методів: 14 базових, варіанти абляцій (variants.py), суфікс -C (децентралізований
    контролер-репліка), суфікс -U (без обізнаності про відмови), підібрані параметри (params_file),
    ручні параметри (method_params);
  * телеметрія (probes.py) без впливу на поведінку методів;
  * завдання групуються за картою (карта будується один раз на завдання), BLAS — 1 потік;
  * у кожному записі: wall_s, in_model (ознаки постраждалих у межах моделі), додаткові метрики
    середовища та контролера, params (якщо не типові), code_hash_base;
  * код виходу 5 — не виконано залежність (наприклад, параметри tune2 ще не остаточні).

Коди виходу: 0 — усе виконано; 1 — помилка (можна перезапускати); 2 — зупинено користувачем;
3 — експеримент уже виконується іншим процесом; 4 — змінено код / конфігурацію / параметри;
5 — не виконано залежність.

Приклади:
  python run_experiments.py --config configs/main.json
  python run_experiments.py --config configs/suite/conf_main.json --workers 3
  python run_experiments.py --config configs/main.json --export-only
  python run_experiments.py --config configs/suite/comm2_R8.json --dry-run
"""
import os
import sys

# Один потік BLAS на процес: паралелізм — лише процесами (інакше на 2-ядерному ПК потоки
# numpy конкурують між собою і з робочими процесами). Має стояти ДО імпорту numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, datetime as dt, hashlib, json, signal, socket, threading, time, zlib  # noqa: E402
import numpy as np                                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Е9 / Е11: сітка іншого розміру ставиться ДО імпорту моделей (змінна оточення успадковується робочими процесами)
_GRID = os.environ.get('SAR_GRID')                                                     # '<size>:s' | '<size>:d'
if _GRID:
    import gridscale                                                                   # noqa: E402
    gridscale.install(int(_GRID.split(':')[0]), scalable=_GRID.endswith(':s'))
from env import MapInstance, SAREnv                                                  # noqa: E402
from registry import BY_NAME, GROUP, expand_methods, resolve, base_name                       # noqa: E402

MODEL_FILES = ['env.py', 'baselines.py', 'novel.py', 'hybrid.py']
EXT_FILES = ['ext_env.py', 'variants.py', 'decentral.py', 'probes.py', 'gridscale.py', 'realmap.py']
SC_OFFSET = {'S1': 0, 'S2': 50000, 'S3': 90000}
STALE_LOCK_S = 120
TASK_RUNS = 60                      # максимум прогонів в одному завданні робочого процесу
EXIT_OK, EXIT_ERR, EXIT_STOP, EXIT_LOCKED, EXIT_CODE_CHANGED, EXIT_DEP = 0, 1, 2, 3, 4, 5
# ключі конфігурації, що не впливають на результати (їх зміна не блокує продовження)
NEUTRAL_KEYS = {'description', 'hypotheses', 'stage', 'priority', 'depends_on', 'analyze', 'note',
                'est_weight', 'paired_with', 'max_workers'}
_stop = False


class DependencyError(Exception):
    """Залежність експерименту ще не готова (код виходу 5)."""


# ------------------------------------------------------------------ службове
def file_hash(files):
    h = hashlib.sha256()
    for f in files:
        with open(os.path.join(HERE, f), 'rb') as fh:
            h.update(fh.read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:16]


def code_hash(extra=()):
    """Хеш 4 файлів моделей + вказаних модулів розширень (у фіксованому порядку)."""
    return file_hash(MODEL_FILES + [f for f in EXT_FILES if f in set(extra)])


def now():
    return dt.datetime.now().isoformat(timespec='seconds')


def atomic_write_json(path, obj, retries=20):
    """Атомарний запис JSON; на Windows os.replace може тимчасово падати (антивірус, читач) — повтор."""
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for k in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    os.replace(tmp, path)


def log(exp_dir, msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with open(os.path.join(exp_dir, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def cfg_signature(cfg):
    """Канонічний підпис конфігурації (без нейтральних ключів) — для перевірки при продовженні."""
    return json.dumps({k: v for k, v in cfg.items() if k not in NEUTRAL_KEYS}, sort_keys=True, ensure_ascii=False)


# ------------------------------------------------------------------ опис експерименту
def method_list(cfg):
    return expand_methods(cfg.get('methods', 'all'))


def env_kind(cfg):
    """base — env.SAREnv (побітово v1); ext — SAREnvExt; dyn — SAREnvDyn; comm — SAREnvComm."""
    if cfg.get('dyn'):
        return 'dyn'
    if cfg.get('comm') or cfg.get('fail') or any(resolve(m)[2] for m in method_list(cfg)):
        return 'comm'
    if cfg.get('row') is not None or cfg.get('rho_out') is not None or cfg.get('survival'):
        return 'ext'
    return 'base'


def required_modules(cfg):
    """Модулі розширень, від яких залежать РЕЗУЛЬТАТИ експерименту (входять у code_hash)."""
    mods = set()
    kind = env_kind(cfg)
    if kind != 'base':
        mods.add('ext_env.py')
    if kind == 'comm':
        mods.add('decentral.py')
    for m in method_list(cfg):
        cls, params, suf = resolve(m)
        core = m[:-2] if suf else m
        if core not in BY_NAME:                          # варіант абляції: визначення — у variants.py
            mods.add('variants.py')
    if cfg.get('params_file'):
        mods.add('variants.py')
    if cfg.get('telemetry'):
        mods.add('probes.py')
    if cfg.get('grid') or cfg.get('realmap'):
        mods.add('gridscale.py')
    if cfg.get('realmap'):
        mods.add('realmap.py')
    return sorted(mods)


def grid_spec(cfg):
    """'<size>:s|d' для експериментів з іншою сіткою / реальною картою, інакше None."""
    g = cfg.get('grid')
    if not g and not cfg.get('realmap'):
        return None
    g = g or {}
    return f"{int(g.get('size', 50))}:{'s' if g.get('scalable', True) else 'd'}"


def _relaunch(want):
    """Перезапуск раннера в новому процесі з потрібною сіткою (модулі моделей уже завантажено з іншою)."""
    import subprocess
    envv = dict(os.environ)
    if want:
        envv['SAR_GRID'] = want
    else:
        envv.pop('SAR_GRID', None)
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # Ctrl+C обробляє дочірній раннер (м'яка зупинка)
    return subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=envv)


def load_tuned(cfg):
    """Підібрані параметри (Е6 v2). -> (dict метод -> параметри, params_hash) або ({}, None)."""
    pf = cfg.get('params_file')
    if not pf:
        return {}, None
    path = pf if os.path.isabs(pf) else os.path.join(HERE, pf)
    if not os.path.exists(path):
        raise DependencyError(f'файл параметрів {pf} ще не створено (спершу завершіть tune2.py)')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not d.get('finalized_at'):
        raise DependencyError(f'параметри у {pf} ще не остаточні (tune2.py не завершено)')
    tuned = {m: v['params'] for m, v in d.get('methods', {}).items()}
    ph = hashlib.sha256(json.dumps(tuned, sort_keys=True).encode()).hexdigest()[:16]
    return tuned, ph


# ------------------------------------------------------------------ план прогонів
def run_key(exp, sc, ms, vd, method):
    return f'{exp}|{sc}|m{ms}|v{vd}|{method}'


def build_plan(cfg):
    methods = method_list(cfg)
    m_lo, m_hi = cfg['maps']
    plan = []
    for sc in cfg['scenarios']:
        for ms in range(m_lo, m_hi + 1):
            for vd in range(cfg['draws']):
                plan.append((sc, ms, vd, list(methods)))
    return plan


# ------------------------------------------------------------------ журнал результатів
def load_journal(path, exp_dir):
    """Читає JSONL, обрізає недописаний хвіст, пропускає пошкоджені рядки. Повертає множину ключів."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'rb') as f:
        data = f.read()
    good_end, pos, bad = 0, 0, 0
    while pos < len(data):
        nl = data.find(b'\n', pos)
        if nl < 0:                                   # останній рядок без \n — недописаний
            break
        line = data[pos:nl]
        try:
            rec = json.loads(line.decode('utf-8'))
            done.add(rec['key'])
            good_end = nl + 1
        except Exception:
            bad += 1
            with open(os.path.join(exp_dir, 'quarantine.jsonl'), 'ab') as q:
                q.write(line + b'\n')
            good_end = nl + 1                        # пошкоджений рядок у середині — ізолюємо
        pos = nl + 1
    if good_end < len(data):
        with open(path, 'r+b') as f:
            f.truncate(good_end)
        log(exp_dir, f'Обрізано недописаний хвіст журналу ({len(data) - good_end} байт) — наслідок аварійного вимкнення')
    if bad:
        log(exp_dir, f'Пошкоджених рядків у журналі: {bad} (перенесено у quarantine.jsonl, будуть перераховані)')
    return done


class Journal:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')

    def append(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ блокування
class Lock:
    def __init__(self, path):
        self.path = path
        self._stop = threading.Event()
        self._thr = None

    def acquire(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding='utf-8') as f:
                    info = json.load(f)
                age = time.time() - info.get('heartbeat', 0)
                if age < STALE_LOCK_S:
                    return False, info
            except Exception:
                pass                                 # пошкоджений lock — вважаємо застарілим
        self.beat()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return True, None

    def _loop(self):
        while not self._stop.wait(10):
            try:
                self.beat()
            except Exception:
                pass

    def beat(self):
        atomic_write_json(self.path, dict(pid=os.getpid(), host=socket.gethostname(), heartbeat=time.time(), time=now()))

    def release(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=5)
        for _ in range(20):
            try:
                os.remove(self.path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.1)


# ------------------------------------------------------------------ фабрики
_MAP_CACHE = {}


def get_map(sc, ms, cfg=None):
    lam = float((cfg or {}).get('lam_len', 5.0))
    real = (cfg or {}).get('realmap')
    key = (sc, ms, lam, real['name'] if real else None)
    if key not in _MAP_CACHE:
        if len(_MAP_CACHE) >= 2:
            _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        if real:
            import realmap
            _MAP_CACHE[key] = realmap.make_instance(real, sc, ms, lam_len=lam)
        elif lam != 5.0:
            _MAP_CACHE[key] = MapInstance(sc, ms, lam_len=lam)
        else:
            _MAP_CACHE[key] = MapInstance(sc, ms)
    return _MAP_CACHE[key]


def apply_survival(cfg):
    """Параметри кривої виживання для цього процесу (типові, якщо в конфігу їх немає)."""
    s = cfg.get('survival')
    if s or 'ext_env' in sys.modules:
        import ext_env as X
        X.apply_env_params(s.get('tau') if s else None, s.get('kappa') if s else None)


def victims_for(cfg, mp, sc, ms, vd):
    """Постраждалі (як у v1) + ознаки «у межах моделі» (1) / «поза моделлю» (0)."""
    rng = np.random.default_rng(1000 * ms + vd + SC_OFFSET[sc])
    rho = cfg.get('rho_out')
    import ext_env as X
    if rho is None:
        v = mp.sample_victims(rng)                      # оригінальна функція — результати як у v1
        v2, flags = X.sample_victims_flags(mp, np.random.default_rng(1000 * ms + vd + SC_OFFSET[sc]))
        if not np.array_equal(v, v2):                    # захист: ознаки мають відповідати тій самій вибірці
            flags = None
        return v, flags
    return X.sample_victims_flags(mp, rng, p_in=round(1.0 - float(rho), 12))


def make_env(cfg, mp, victims, seed, inst):
    M, T = cfg.get('n_agents', 5), cfg.get('T', 300)
    kind = env_kind(cfg)
    if kind == 'base':
        return SAREnv(mp, victims, seed=seed, M=M, T=T)
    import ext_env as X
    row = cfg.get('row')
    if kind == 'dyn':
        d = cfg['dyn']
        return X.SAREnvDyn(mp, victims, seed, M=M, T=T, row=row, key=inst,
                           p_move=float(d.get('p_move', 0.0)), attract_eps=float(d.get('attract_eps', 1.0)))
    if kind == 'comm':
        return X.SAREnvComm(mp, victims, seed, M=M, T=T, row=row, key=inst, fail=cfg.get('fail'))
    return X.SAREnvExt(mp, victims, seed, M=M, T=T, row=row, key=inst)


def method_params(cfg, name, tuned):
    """-> (клас, параметри, суфікс) з урахуванням варіанта, підібраних і ручних параметрів."""
    cls, params, suf = resolve(name)
    core = name[:-2] if suf else name                   # назва без -C / -U
    if tuned and core in tuned:
        import variants
        cls = variants.TUNABLE.get(core, cls)
        params.update(tuned[core])
    params.update(cfg.get('method_params', {}).get(core, {}))
    return cls, params, suf


def make_method(cfg, name, tuned):
    cls, params, suf = method_params(cfg, name, tuned)
    kind = env_kind(cfg)
    if suf == '-C' or (kind == 'comm' and suf != '-U'):
        from decentral import CommController
        comm = cfg.get('comm') or {}
        mode = 'dec' if suf == '-C' else 'oracle'
        return CommController(cls, params, mode=mode, R=comm.get('R'), base_node=comm.get('base_node', True),
                              name=name), params
    m = cls()
    for k, v in params.items():
        setattr(m, k, v)
    return m, params


# ------------------------------------------------------------------ виконання
def run_single(cfg, sc, ms, vd, mname, ctx):
    """Один прогін. ctx: code_hash, code_hash_base, tuned (dict), params_hash."""
    w0 = time.perf_counter()
    apply_survival(cfg)
    mp = get_map(sc, ms, cfg)
    victims, flags = victims_for(cfg, mp, sc, ms, vd)
    inst = f'{sc}-m{ms}-v{vd}'
    env = make_env(cfg, mp, victims, zlib.crc32(inst.encode()), inst)
    m, params = make_method(cfg, mname, ctx.get('tuned') or {})
    if hasattr(mp, 'begin_run'):                      # Е9: власний облік часу на відстані / рівняння
        mp.begin_run()
    m.reset(env, np.random.default_rng(7 * ms + vd))
    if hasattr(mp, 'begin_run'):
        mp.begin_run(clear=False)                     # час reset() не входить у ms_per_step
    probe = None
    if cfg.get('telemetry'):
        import probes
        probe = probes.attach(m)
    t_act = 0.0
    traj_maps = cfg.get('traj_maps')
    save_traj = bool(cfg.get('save_trajectories', False)) and (traj_maps is None or ms in traj_maps)
    traj = [env.pos.copy()] if save_traj else None
    while not env.done():
        t0 = time.perf_counter()
        a = m.act(env)
        t_act += time.perf_counter() - t0
        if probe is not None:
            probe.after_act(env)
        env.step(a)
        if save_traj:
            traj.append(env.pos.copy())
    met, cp = env.metrics()
    rec = dict(key=run_key(cfg['name'], sc, ms, vd, mname), exp=cfg['name'], scenario=sc, map=ms, draw=vd,
               instance=inst, method=mname, group=GROUP[mname])
    rec.update({k: float(v) for k, v in met.items()})
    rec['ms_per_step'] = 1000 * t_act / env.T
    rec['found_t'] = [int(x) for x in env.found_t]          # момент виявлення кожного постраждалого (-1 — не знайдено)
    rec['victims'] = [int(x) for x in victims]              # клітинки постраждалих (початкові)
    rec['path_len'] = int(env.path_len)
    if flags is not None:
        rec['in_model'] = [int(x) for x in flags]           # 1 — постраждалий з апріору, 0 — поза моделлю
    if params:
        rec['params'] = jsonable(params)
    if cfg.get('survival'):
        import env as E
        rec['tau_s'], rec['kappa_s'] = float(E.TAU_S), float(E.KAPPA_S)
    if hasattr(mp, 'run_counters'):
        rec.update(jsonable(mp.run_counters(env.T)))
    if cfg.get('grid') or cfg.get('realmap'):
        import env as E
        rec['grid'] = int(E.W)
        if cfg.get('realmap'):
            rec['realmap'] = cfg['realmap']['name']
    if hasattr(env, 'extra_metrics'):
        rec.update(jsonable(env.extra_metrics()))
    if hasattr(m, 'extra_metrics'):
        rec.update(jsonable(m.extra_metrics()))
    if probe is not None:
        rec.update(jsonable(probe.fields()))
    if save_traj:                                           # траєкторії: (T+1) x M індексів клітинок
        tdir = os.path.join(HERE, 'results', cfg['name'], 'traj')
        os.makedirs(tdir, exist_ok=True)
        base_fn = os.path.join(tdir, rec['key'].replace('|', '__').replace('*', 'x'))
        arrs = {'': np.array(traj, dtype=np.int16)}
        if getattr(env, 'p_move', 0) > 0:                   # рухомі постраждалі — також їхні траєкторії
            arrs['_victims'] = env.victim_traj.astype(np.int16)
        for suf, arr in arrs.items():
            fn = base_fn + suf + '.npy'
            with open(fn + '.tmp', 'wb') as fh:
                np.save(fh, arr)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(fn + '.tmp', fn)
            rec['traj_file' + suf] = os.path.relpath(fn, HERE)
    rec['cpos'] = [round(float(x), 5) for x in cp]
    rec['code_hash'] = ctx['code_hash']
    rec['code_hash_base'] = ctx['code_hash_base']
    if ctx.get('params_hash'):
        rec['params_hash'] = ctx['params_hash']
    rec['wall_s'] = round(time.perf_counter() - w0, 4)
    rec['ts'] = now()
    return rec


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # зупинкою керує головний процес
    try:                                             # на Linux/macOS (fork) робочий процес успадкував би
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # обробник SIGTERM головного і не завершувався б
    except (AttributeError, ValueError):             # на Pool.terminate() -> зависання
        pass


def run_task(args):
    """Завдання для робочого процесу: невиконані прогони однієї карти (кілька розстановок)."""
    cfg, sc, ms, items, ctx = args
    out = []
    for vd, methods in items:
        for m in methods:
            out.append(run_single(cfg, sc, ms, vd, m, ctx))
    return out


def _on_signal(signum, frame):
    global _stop
    _stop = True
    print('\n>>> Отримано сигнал зупинки: завершую поточне завдання і виходжу (дані збережено).', flush=True)


def export(exp_dir):
    import pandas as pd
    recs = []
    with open(os.path.join(exp_dir, 'runs.jsonl'), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    df = pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items() if k != 'cpos'} for r in recs])
    df.to_csv(os.path.join(exp_dir, 'results.csv'), index=False, encoding='utf-8-sig')
    curves = {}
    for r in recs:
        curves.setdefault(f"{r['scenario']}|{r['method']}", []).append(r['cpos'])
    np.savez_compressed(os.path.join(exp_dir, 'cpos_curves.npz'),
                        **{k: np.array(v, dtype=np.float32) for k, v in curves.items()})
    return len(df)


def make_tasks(cfg, done, ctx):
    by_map = {}
    for sc, ms, vd, methods in build_plan(cfg):
        todo = [m for m in methods if run_key(cfg['name'], sc, ms, vd, m) not in done]
        if todo:
            by_map.setdefault((sc, ms), []).append((vd, todo))
    tasks = []
    for (sc, ms), items in by_map.items():
        chunk, n = [], 0
        for vd, todo in items:
            chunk.append((vd, todo))
            n += len(todo)
            if n >= TASK_RUNS:
                tasks.append((cfg, sc, ms, chunk, ctx))
                chunk, n = [], 0
        if chunk:
            tasks.append((cfg, sc, ms, chunk, ctx))
    return tasks


def main():
    try:                                             # консоль Windows (cp1251/cp866) не має частини символів
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='лише показати план, модулі й хеш')
    ap.add_argument('--allow-code-change', action='store_true',
                    help='продовжити попри зміну коду / конфігурації (НЕ рекомендовано для дисертації)')
    a = ap.parse_args()
    with open(a.config, encoding='utf-8') as f:
        cfg = json.load(f)
    want = grid_spec(cfg)
    if want != os.environ.get('SAR_GRID'):
        return _relaunch(want)
    exp_dir = os.path.join(HERE, 'results', cfg['name'])
    mods = required_modules(cfg)
    if cfg.get('max_workers'):
        a.workers = max(1, min(a.workers, int(cfg['max_workers'])))
    if a.dry_run:
        total = sum(len(p[3]) for p in build_plan(cfg))
        print(json.dumps(dict(name=cfg['name'], env=env_kind(cfg), runs=total, modules=mods,
                              code_hash=code_hash(mods), code_hash_base=code_hash(),
                              methods=method_list(cfg)), ensure_ascii=False, indent=1))
        return EXIT_OK
    os.makedirs(exp_dir, exist_ok=True)
    jpath = os.path.join(exp_dir, 'runs.jsonl')
    if a.export_only:
        print('Експортовано записів:', export(exp_dir))
        return EXIT_OK
    if cfg.get('realmap'):
        import realmap
        err = realmap.check(cfg['realmap'])
        if err:
            log(exp_dir, f'ОЧІКУВАННЯ ЗАЛЕЖНОСТІ: {err}.')
            return EXIT_DEP
    try:
        tuned, phash = load_tuned(cfg)
    except DependencyError as e:
        log(exp_dir, f'ОЧІКУВАННЯ ЗАЛЕЖНОСТІ: {e}. Експеримент буде запущено пізніше.')
        return EXIT_DEP

    lock = Lock(os.path.join(exp_dir, 'run.lock'))
    ok, info = lock.acquire()
    if not ok:
        print(f'Експеримент уже виконується (pid={info.get("pid")}, host={info.get("host")}). '
              f'Якщо це не так — зачекайте {STALE_LOCK_S} с або видаліть run.lock.')
        return EXIT_LOCKED
    try:
        chash, chash_base = code_hash(mods), code_hash()
        snap_path = os.path.join(exp_dir, 'config.json')
        if os.path.exists(snap_path):
            with open(snap_path, encoding='utf-8') as f:
                snap = json.load(f)
            if snap['code_hash'] != chash and not a.allow_code_change:
                log(exp_dir, f'ЗУПИНКА: код моделей змінено ({snap["code_hash"]} -> {chash}). '
                             f'Створіть новий експеримент (інше "name" у конфігу) або запустіть з --allow-code-change.')
                return EXIT_CODE_CHANGED
            if 'config' in snap and cfg_signature(snap['config']) != cfg_signature(cfg) and not a.allow_code_change:
                log(exp_dir, 'ЗУПИНКА: конфігурацію експерименту змінено після старту (результати змішалися б). '
                             'Поверніть попередній конфіг або дайте новий "name".')
                return EXIT_CODE_CHANGED
            if snap.get('params_hash') != phash and not a.allow_code_change:
                log(exp_dir, f'ЗУПИНКА: підібрані параметри змінилися ({snap.get("params_hash")} -> {phash}).')
                return EXIT_CODE_CHANGED
        else:
            import platform, scipy
            atomic_write_json(snap_path, dict(config=cfg, code_hash=chash, code_hash_base=chash_base, modules=mods,
                                              params_hash=phash, tuned_params=tuned or None, created=now(),
                                              runner='v2', env_kind=env_kind(cfg),
                                              python=sys.version.split()[0], numpy=np.__version__,
                                              scipy=scipy.__version__, os=platform.platform(),
                                              cpu=platform.processor() or platform.machine(),
                                              cpu_count=os.cpu_count(), host=socket.gethostname()))
        done = load_journal(jpath, exp_dir)
        total = sum(len(p[3]) for p in build_plan(cfg))
        ctx = dict(code_hash=chash, code_hash_base=chash_base, tuned=tuned, params_hash=phash)
        tasks = make_tasks(cfg, done, ctx)
        n_todo = sum(len(ms_) for t in tasks for _, ms_ in t[3])
        log(exp_dir, f'Старт: {cfg["name"]} | виконано {total - n_todo}/{total} | залишилось {n_todo} | '
                     f'workers={a.workers} | code={chash} | модулі: {", ".join(mods) or "лише моделі"}')
        if n_todo == 0:
            export(exp_dir)
            log(exp_dir, 'Усі прогони виконано. Результати експортовано у results.csv та cpos_curves.npz')
            return EXIT_OK

        signal.signal(signal.SIGINT, _on_signal)
        try:
            signal.signal(signal.SIGTERM, _on_signal)
        except (AttributeError, ValueError):
            pass
        stop_file = os.path.join(exp_dir, 'STOP')
        journal = Journal(jpath)
        t0, n_new = time.time(), 0
        prog_path = os.path.join(exp_dir, 'progress.json')

        def record(rec):
            nonlocal n_new
            journal.append(rec)
            n_new += 1
            el = time.time() - t0
            rate = n_new / el if el > 0 else 0
            left = n_todo - n_new
            if n_new % 5 == 0 or left == 0:
                atomic_write_json(prog_path, dict(experiment=cfg['name'], done=total - left, total=total,
                                                  percent=round(100 * (total - left) / total, 2),
                                                  session_runs=n_new, runs_per_min=round(60 * rate, 1),
                                                  eta_min=round(left / rate / 60, 1) if rate else None,
                                                  last=rec['key'], updated=now()))

        def should_stop():
            return _stop or os.path.exists(stop_file)

        try:
            if a.workers <= 1:
                for (cfg_, sc, ms, items, ctx_) in tasks:
                    for vd, todo in items:
                        for m in todo:
                            if should_stop():
                                raise KeyboardInterrupt
                            record(run_single(cfg_, sc, ms, vd, m, ctx_))
                        print(f'  {sc}-m{ms}-v{vd}: готово ({total - n_todo + n_new}/{total})', flush=True)
            else:
                import multiprocessing as mpc
                with mpc.Pool(a.workers, initializer=_worker_init) as pool:
                    for recs in pool.imap_unordered(run_task, tasks):
                        for r in recs:
                            record(r)
                        print(f'  {recs[0]["scenario"]}-m{recs[0]["map"]}: готово '
                              f'({total - n_todo + n_new}/{total})', flush=True)
                        if should_stop():
                            pool.terminate()
                            raise KeyboardInterrupt
        except KeyboardInterrupt:
            journal.close()
            if os.path.exists(stop_file):
                os.remove(stop_file)
            log(exp_dir, f'Зупинено користувачем. За сесію збережено {n_new} прогонів. '
                         f'Повторний запуск продовжить з місця зупинки.')
            return EXIT_STOP
        journal.close()
        log(exp_dir, f'Усі {total} прогонів записано в runs.jsonl; експорт results.csv і cpos_curves.npz...')
        export(exp_dir)
        log(exp_dir, f'Експеримент завершено: {total} прогонів. Експорт: results.csv, cpos_curves.npz')
        return EXIT_OK
    except Exception as e:
        import traceback
        log(exp_dir, f'ПОМИЛКА: {type(e).__name__}: {e}. Дані до помилки збережено; перезапустіть раннер.\n'
                     + traceback.format_exc())
        raise
    finally:
        lock.release()


if __name__ == '__main__':
    sys.exit(main())
