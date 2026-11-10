# -*- coding: utf-8 -*-
"""
Обмежений зв'язок і відмови агентів — модель v2 (Е4 v2).

ЗНАННЯ АГЕНТА (точне злиття без подвійного врахування).
Байєсівське оновлення SAREnv  q' = q(1-p)/(1-qp)  у шансах o = q/(1-q) є множенням:  o' = o(1-p).
Отже карта переконань — функція лише початкових шансів і СУМИ свідчень log(1-p) усіх оглядів:
    q(c) = o0(c) e^{S(c)} / (1 + o0(c) e^{S(c)}),   S(c) = sum_j E_j(c),
де E_j — накопичені свідчення агента-джерела j (рядок j). Свідчення кожного джерела ростуть
монотонно й однозначно визначаються номером версії (кроком) v_j, тому знання вузла — це лише
вектор версій ver[j], множина відомих виявлень і відомих відмов. Злиття в компоненті зв'язності:
    ver = max по вузлах,  found = max,  dead = max
— ідемпотентне, комутативне й асоціативне; кожен огляд враховується рівно один раз
(на відміну від злиття мінімумом у v1, яке є лише наближенням).

КОНТРОЛЕР-РЕПЛІКА (replicated controller).
Кожен агент виконує власну копію (репліку) ройового методу для всієї команди на власних знаннях:
члени його компоненти — у поточних позиціях, решта агентів — «привиди» в останніх відомих позиціях.
Агент виконує лише свою дію. У межах компоненти репліки синхронізуються (усі беруть репліку
агента з найменшим номером), тож внутрішня координація методу (послідовне виснаження GREEDY,
заборона клітинок HEDAC*, гра розподілу SIFD, наміри PAS, ролі BSS) зберігається всередині компоненти.
Відомі відмови вилучають агента з команди репліки (перебудова команди з перенесенням стану).

ВЛАСТИВІСТЬ-КОНТРОЛЬ: за R = inf і без відмов одна компонента, одна репліка, знання повні
=> дії побітово збігаються з централізованим методом у SAREnv (перевіряє тест і експеримент comm2_Rinf).

Режими:  mode='oracle' — миттєво злиті знання (але обізнаність про відмови: команда = живі агенти);
         mode='dec'    — децентралізовані знання з радіусом зв'язку R.
"""
import copy
import zlib
import numpy as np

from env import N, RC, SAREnv
import novel as NV

# атрибути методів, що мають по одному елементу на агента (для перебудови команди після відмови)
PER_AGENT_ATTRS = {'wps', 'k', 'dir', 'left', 'x', 'v', 'pbest', 'prev', 'assign', 'plans', 'skeptic',
                   'tgt', 'mode', 'levy_tgt', 'hist', 'heading', 'vdir', 'sweep_wp'}


class LocalView:
    """Подання середовища для репліки: те, що відомо компоненті зв'язності."""
    gain_map = SAREnv.gain_map
    gain_at = SAREnv.gain_at

    def __init__(self, mp, M, T, t, q, q_total0, pos, found_t, victims, obs_events=None):
        self.mp, self.M, self.T, self.t = mp, M, T, t
        self.q, self.q_total0, self.pos = q, q_total0, pos
        self.found_t, self.victims = found_t, victims
        self.obs_events = obs_events


class BSSCMixin:
    """
    BSS для реплік: якщо подання містить obs_events (неповні знання / злиття з минулого), свідчення
    моделей оновлюються за подіями «огляд агента j на кроці s» у канонічному порядку (s, j);
    інакше — оригінальний BSS.observe (побітова рівність з оракулом).
    """

    def act(self, env):
        ev = getattr(env, 'obs_events', None)
        if ev is not None:
            self._observe_events(env.mp, ev)
            self.seen_t = env.t
        return super().act(env)

    def _observe_events(self, mp, events):
        i = 0
        while i < len(events):
            t_e = events[i][0]
            j = i
            newly = set()
            while j < len(events) and events[j][0] == t_e:
                newly |= events[j][3]
                j += 1
            for (_, _, cell, _) in events[i:j]:
                fi, fp = mp.fp_idx[cell], mp.fp_p[cell]
                ok = fi < N
                fi, fp = fi[ok], fp[ok]
                det = np.array([c in newly for c in fi])
                for m in range(2):
                    q = self.qm[m][fi]
                    lik = np.where(det, q * fp, 1 - q * fp)
                    self.logw[m] += np.log(np.maximum(lik, 1e-300)).sum()
                    self.qm[m][fi] = np.where(det, 0.0, q * (1 - fp) / (1 - q * fp))
                for c in fi[det]:
                    newly.discard(c)
            i = j
        self.logw -= self.logw.max()


def dec_class(cls):
    """Клас репліки: для BSS і похідних — з обробкою подій спостережень."""
    if issubclass(cls, NV.BSS):
        return type(cls.__name__ + '_C', (BSSCMixin, cls), {})
    return cls


class Replica:
    __slots__ = ('m', 'team', 'members', 'proc_ver')

    def __init__(self, m, team, members, proc_ver):
        self.m, self.team, self.members, self.proc_ver = m, tuple(team), frozenset(members), proc_ver


def _select(val, idx):
    if isinstance(val, np.ndarray):
        return val[idx].copy()
    if isinstance(val, list):
        return [copy.deepcopy(val[k]) for k in idx]
    return copy.deepcopy(val)


def reform(rep, new_team):
    """Перебудова репліки для меншої команди (відомі відмови) з перенесенням стану методу."""
    old = rep.m
    idx = [rep.team.index(j) for j in new_team]
    new = old.__class__.__new__(old.__class__)
    for a, val in old.__dict__.items():
        if a == 'rng':
            new.rng = old.rng
        elif a in PER_AGENT_ATTRS:
            setattr(new, a, _select(val, idx))
        else:
            setattr(new, a, copy.deepcopy(val))
    resync(new)
    return Replica(new, new_team, rep.members, rep.proc_ver.copy())


def _merge_global(dst, src):
    """Злиття глобального стану двох реплік: феромон — поелементний максимум, відомі клітинки — об'єднання."""
    for a in ('tau',):
        if isinstance(getattr(dst, a, None), np.ndarray) and isinstance(getattr(src, a, None), np.ndarray):
            setattr(dst, a, np.maximum(getattr(dst, a), getattr(src, a)))
    if isinstance(getattr(dst, 'known', None), set) and isinstance(getattr(src, 'known', None), set):
        dst.known = dst.known | src.known


def adopt_rows(rep, src, i):
    """Агент i — авторитетне джерело власного стану: переносимо його рядки з репліки src у rep."""
    if i not in src.team or i not in rep.team:
        return
    a_dst, a_src = rep.team.index(i), src.team.index(i)
    for a in PER_AGENT_ATTRS:
        if a in src.m.__dict__ and a in rep.m.__dict__:
            vs, vd = src.m.__dict__[a], rep.m.__dict__[a]
            if isinstance(vd, np.ndarray):
                vd[a_dst] = vs[a_src]
            elif isinstance(vd, list):
                vd[a_dst] = copy.deepcopy(vs[a_src])
    _merge_global(rep.m, src.m)


def resync(m):
    """Після зміни складу компоненти: примусове перепланування / перерозподіл."""
    if hasattr(m, 'plans') and isinstance(m.plans, list):
        m.plans = [[] for _ in m.plans]
    if hasattr(m, 'assign') and isinstance(m.assign, np.ndarray):
        m.assign[:] = -1


class CommController:
    """Контролер рою з обмеженим зв'язком / відмовами. Інтерфейс як у baselines.Method."""

    def __init__(self, inner_cls, params=None, mode='dec', R=float('inf'), base_node=True, name=None):
        self.inner_cls = dec_class(inner_cls) if mode == 'dec' else inner_cls
        self.params = dict(params or {})
        self.mode = mode
        self.R = float('inf') if R is None else float(R)
        self.base_node = bool(base_node)
        self.name = name or inner_cls.name

    # ---------------------------------------------------------------- службове
    def _new_inner(self):
        m = self.inner_cls()
        for k, v in self.params.items():
            setattr(m, k, v)
        return m

    def _alive(self, env):
        return env.alive.copy() if hasattr(env, 'alive') else np.ones(env.M, bool)

    def _components(self, env, alive):
        """Компоненти графа зв'язності: вузли 0..M-1 — живі агенти, вузол M — база."""
        M = env.M
        nodes = [i for i in range(M) if alive[i]] + ([M] if self.base_node else [])
        if not np.isfinite(self.R):
            return [nodes]
        xy = {i: RC[env.pos[i]] for i in range(M) if alive[i]}
        if self.base_node:
            xy[M] = RC[env.mp.base]
        R2 = self.R * self.R
        comp, seen = [], set()
        for s in nodes:
            if s in seen:
                continue
            stack, cur = [s], []
            seen.add(s)
            while stack:
                u = stack.pop()
                cur.append(u)
                for w in nodes:
                    if w not in seen and ((xy[u] - xy[w]) ** 2).sum() <= R2:
                        seen.add(w)
                        stack.append(w)
            comp.append(sorted(cur))
        return comp

    # ---------------------------------------------------------------- життєвий цикл
    def reset(self, env, rng):
        M, T = env.M, env.T
        self.M = M
        self.seed0 = zlib.crc32((getattr(env, 'key', '') + '|ctl').encode())
        self.q0 = env.q.copy()
        with np.errstate(divide='ignore'):
            self.odds0 = self.q0 / (1.0 - self.q0)
        self.E_hist = [[np.zeros(N)] for _ in range(M)]      # свідчення джерела j за версіями 0..v
        self.traj = np.full((M, T + 1), -1, int)
        self.traj[:, 0] = env.pos
        self.gver = np.zeros(M, int)                           # найновіша версія кожного джерела
        nK = len(env.victims)
        self.kver = np.zeros((M + 1, M), int)                  # вузли: агенти + база
        self.kfound = np.full((M + 1, nK), -1, int)
        self.kdead = np.full((M + 1, M), -1, int)
        self.prev_comp = [list(range(M)) + ([M] if self.base_node else [])]
        inner = self._new_inner()
        view0 = LocalView(env.mp, M, T, env.t, env.q.copy(), env.q_total0, env.pos.copy(),
                          env.found_t, env.victims)
        inner.reset(view0, rng)
        rep = Replica(inner, range(M), range(M), np.zeros(M, int))
        self.rep_of = [rep] * M
        self.acc = dict(disc=0.0, disc_n=0, age=0.0, age_n=0, merges=0, comps=0.0, reps=0.0, steps=0, reforms=0)

    def _ingest(self, env, alive):
        """Власні огляди кроку t (для живих агентів) і відомі відмови."""
        t, M = env.t, env.M
        if t == 0:
            return
        mp = env.mp
        for i in range(M):
            if alive[i]:
                row = self.E_hist[i][-1].copy()
                fi, fp = mp.fp_idx[env.pos[i]], mp.fp_p[env.pos[i]]
                ok = fi < N
                row[fi[ok]] += np.log1p(-fp[ok])
                self.E_hist[i].append(row)
                self.gver[i] = t
                self.traj[i, t] = env.pos[i]
                self.kver[i, i] = t
            else:
                self.traj[i, t] = env.pos[i]
        if hasattr(env, 'finder'):
            for v in np.nonzero(env.found_t == t)[0]:
                f = env.finder[v]
                if f >= 0:
                    self.kfound[f, v] = t
        else:                                                  # SAREnv без відмов: знають усі, хто бачив
            for v in np.nonzero(env.found_t == t)[0]:
                for i in range(M):
                    if env.victims[v] in mp.fp_idx[env.pos[i]]:
                        self.kfound[i, v] = t
        dt = getattr(env, 'death_t', None)
        if dt is not None:
            newly = np.nonzero(dt == t)[0]
            for j in newly:                                    # втрату «серцебиття» помічає компонента j
                for comp in self.prev_comp:
                    if j in comp:
                        for node in comp:
                            self.kdead[node, j] = t

    def _belief(self, ver, found, env):
        S = np.zeros(N)
        for j in range(self.M):
            S += self.E_hist[j][ver[j]]
        o = self.odds0 * np.exp(S)
        q = o / (1.0 + o)
        q[~np.isfinite(q)] = 0.0
        fv = found >= 0
        if fv.any():
            q[env.victims[fv]] = 0.0
        return q

    def act(self, env):
        M, t = env.M, env.t
        alive = self._alive(env)
        if self.mode == 'oracle':
            return self._act_oracle(env, alive)
        self._ingest(env, alive)
        comps = self._components(env, alive)
        # --- злиття знань у компонентах
        for comp in comps:
            if len(comp) < 2:
                continue
            nv, nf, nd = self.kver[comp].max(0), self.kfound[comp].max(0), self.kdead[comp].max(0)
            for node in comp:
                if (self.kver[node] != nv).any() or (self.kfound[node] != nf).any() or (self.kdead[node] != nd).any():
                    self.acc['merges'] += 1
                self.kver[node], self.kfound[node], self.kdead[node] = nv, nf, nd
        self.prev_comp = comps
        # --- метрики зв'язку
        n_alive = int(alive.sum())
        if self.base_node and n_alive:
            base_comp = next(c for c in comps if M in c)
            self.acc['disc'] += sum(1 for i in range(M) if alive[i] and i not in base_comp) / n_alive
            self.acc['disc_n'] += 1
        ages = [t - self.kver[i, j] for i in range(M) if alive[i] for j in range(M)
                if j != i and self.kdead[i, j] < 0]
        if ages:
            self.acc['age'] += float(np.mean(ages)); self.acc['age_n'] += 1
        agent_comps = [[i for i in c if i < M] for c in comps]
        agent_comps = [c for c in agent_comps if c]
        self.acc['comps'] += len(agent_comps); self.acc['steps'] += 1
        # --- рішення кожної компоненти
        dead_true = getattr(env, 'death_t', np.full(M, 10 ** 9)) <= t
        out = env.pos.copy()
        used = 0
        for comp in sorted(agent_comps, key=min):
            lead = comp[0]
            orig = self.rep_of[lead]
            rep = orig
            if any(orig is self.rep_of[j] for j in range(M) if alive[j] and j not in comp):
                rep = copy.deepcopy(orig)
                # копія репліки після розщеплення компоненти — власний незалежний генератор
                # (інакше розділені агенти «ділили» б один випадковий потік); детермінований ключем
                rep.m.rng = np.random.default_rng([self.seed0, int(t), int(lead)])
            merged_from = set()
            for i in comp[1:]:                                 # власний стан членів з інших реплік
                src = self.rep_of[i]
                if src is not orig and src is not rep and id(src) not in merged_from:
                    for j in comp:
                        if self.rep_of[j] is src:
                            adopt_rows(rep, src, j)
                    merged_from.add(id(src))
            team = tuple(j for j in range(M) if self.kdead[lead, j] < 0)
            if team != rep.team:
                rep = reform(rep, team)
                self.acc['reforms'] += 1
            if rep.members != frozenset(comp):
                resync(rep.m)
                rep.members = frozenset(comp)
            kv, kf = self.kver[lead], self.kfound[lead]
            complete = ((kv == self.gver).all() and (kf == env.found_t).all()
                        and ((self.kdead[lead] >= 0) == dead_true).all())
            q = env.q.copy() if complete else self._belief(kv, kf, env)
            pos = np.array([env.pos[j] if j in comp else self.traj[j, kv[j]] for j in team], int)
            found_v = np.where(kf >= 0, kf, -1)
            vict_v = np.where(kf >= 0, env.victims, -1)
            events = None
            if isinstance(rep.m, BSSCMixin):
                ev = []
                for j in range(M):
                    for s in range(rep.proc_ver[j] + 1, kv[j] + 1):
                        det = {int(env.victims[v]) for v in np.nonzero(kf == s)[0]}
                        ev.append((s, j, int(self.traj[j, s]), det))
                ev.sort(key=lambda e: (e[0], e[1]))
                # стандартний шлях BSS.observe (побітово як оракул): нові лише огляди кроку t
                # усіх членів команди, які всі в компоненті, і знання повні
                standard = (not ev) or (complete and set(comp) == set(team)
                                        and all(e[0] == t for e in ev) and [e[1] for e in ev] == list(team))
                events = None if standard else ev
            view = LocalView(env.mp, len(team), env.T, t, q, env.q_total0, pos, found_v, vict_v, events)
            a = rep.m.act(view)
            rep.proc_ver = kv.copy()
            used += 1
            for i in comp:
                out[i] = a[team.index(i)]
                self.rep_of[i] = rep
        self.acc['reps'] += used
        return out

    def _act_oracle(self, env, alive):
        """Оракул: спільна карта SAREnv, команда = живі агенти (обізнаність про відмови)."""
        M = env.M
        team = tuple(i for i in range(M) if alive[i])
        if not team:                                   # відмовили всі агенти: рій нерухомий, метод не викликається
            self.acc['steps'] += 1
            return env.pos.copy()
        rep = self.rep_of[0]
        if team != rep.team:
            rep = reform(rep, team)
            self.acc['reforms'] += 1
            self.rep_of = [rep] * M
        view = LocalView(env.mp, len(team), env.T, env.t, env.q.copy(), env.q_total0,
                         env.pos[list(team)].copy(), env.found_t, env.victims)
        a = rep.m.act(view)
        out = env.pos.copy()
        for k, i in enumerate(team):
            out[i] = a[k]
        self.acc['steps'] += 1
        self.acc['comps'] += 1
        self.acc['reps'] += 1
        return out

    def extra_metrics(self):
        a = self.acc
        st = max(1, a['steps'])
        return dict(frac_disconnected=(a['disc'] / a['disc_n']) if a['disc_n'] else 0.0,
                    info_age=(a['age'] / a['age_n']) if a['age_n'] else 0.0,
                    n_merges=int(a['merges']), n_components=a['comps'] / st,
                    n_replicas=a['reps'] / st, n_reforms=int(a['reforms']))
