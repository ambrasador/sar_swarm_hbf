# -*- coding: utf-8 -*-
"""
Стаття 1: механічний відбір остаточної конфігурації HBF і значення ε* конкурента HEDAC*-ε за результатами
розроблення та фіксація протоколу підтвердження ДО підтверджувального прогону.

Вхід (тека dev/): dev_r1.json (карти 100–111), dev_r3b.json (конфігурації раунду 1 на картах 112–119),
dev_r2.json, dev_r3.json, dev_r4a.json (карти 100–119), dev_r4b.json (кандидати етапу 2 на картах 120–159).
Вихід: frozen.json, dev/dev_summary.json, PROTOCOL_S1.md (+ PROTOCOL_S1.sha256).

Правило (dev/STAGE2_RULE.md): кандидати — 5 найкращих конфігурацій HBF і 3 найкращі ε за J̄ на картах 100–119;
остаточний вибір — максимум J̄ на всіх 60 картах розроблення 100–159 (1680 екземплярів), окремо для HBF і ε.
"""
import datetime as dt, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, 'dev')
sys.path.insert(0, HERE); sys.path.insert(0, DEV)
import configs_dev as CD  # noqa: E402

COND = ['S1|None', 'S2|None', 'S3|0.0', 'S3|0.2', 'S3|0.4', 'S3|0.6', 'S3|0.8']


def load(n):
    return json.load(open(os.path.join(DEV, n), encoding='utf-8'))


def merge(files):
    data = {}
    for f in files:
        for name, d in load(f).items():
            for c, inst in d.items():
                dd = data.setdefault(name, {}).setdefault(c, {})
                for i, v in inst.items():
                    if i in dd and abs(dd[i] - v) > 1e-12:
                        raise SystemExit(f'розбіжність між раундами: {name} {c} {i}')
                    dd[i] = v
    return data


def summarize(data, n_per):
    rows = {}
    for name, d in data.items():
        if any(len(d.get(c, {})) != n_per for c in COND):
            continue
        per = {c: float(np.mean(list(d[c].values()))) for c in COND}
        rows[name] = dict(per=per, Jbar=float(np.mean(list(per.values()))))
    eps = [k for k in rows if k.startswith('H-e')]
    best = {c: max(rows[k]['per'][c] for k in eps) for c in COND}
    for r in rows.values():
        r['R'] = float(max(best[c] - r['per'][c] for c in COND))
    return rows


tune = summarize(merge(['dev_r1.json', 'dev_r3b.json', 'dev_r2.json', 'dev_r3.json', 'dev_r4a.json']), 80)
cand = load('stage2_candidates.json')
full = summarize(merge(['dev_r1.json', 'dev_r3b.json', 'dev_r2.json', 'dev_r3.json', 'dev_r4a.json', 'dev_r4b.json']), 240)
val = summarize(merge(['dev_r4b.json']), 160)


def keyf(k, rows):
    c = CD.CFG[k]
    w = c.get('w_rest', 1.0 if c.get('prior_w') else 0.0)
    return (-round(rows[k]['Jbar'], 9), len(c['rhos']), w)


hbf_star = sorted(cand['hbf'], key=lambda k: keyf(k, full))[0]
eps_star = sorted(cand['eps'], key=lambda k: keyf(k, full))[0]
cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CD.CFG[hbf_star].items()}
cfg.setdefault('beta', 1.0)
if 'prior_w' not in cfg:
    cfg.setdefault('w_rest', 0.5)
eps = CD.CFG[eps_star]['rhos'][0]
# сімейства чутливості (вторинний аналіз): інші розумні апріорні налаштування HBF і сітка ε
SENS_HBF = [k for k in ['FG3-w30', 'FG3-w70', 'FG5-w50', 'F2-w30', 'FG5-w30'] if k != hbf_star][:4]
SENS_EPS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
n_hbf = len([k for k in tune if k in CD.CFG and not k.startswith('H-e') and not CD.is_ablation(k)])
n_eps = len([k for k in tune if k.startswith('H-e')])
frozen = dict(hbf_name=hbf_star, hbf=cfg, eps_name=eps_star, eps_star=eps, eps_tag=f'{int(round(eps * 100)):02d}',
              sens_hbf={k: {a: (list(b) if isinstance(b, tuple) else b) for a, b in CD.CFG[k].items()} for k in SENS_HBF},
              sens_eps=SENS_EPS,
              rule='кандидати — 5 найкращих HBF і 3 найкращі ε за J̄ на картах 100–119; вибір — максимум J̄ на картах 100–159 (1680 екземплярів)',
              n_hbf_configs=n_hbf, n_eps_values=n_eps,
              dev={k: dict(J_100_119=tune[k]['Jbar'], R_100_119=tune[k]['R'], J_120_159=val[k]['Jbar'] if k in val else None,
                           J_100_159=full[k]['Jbar'] if k in full else None) for k in tune},
              frozen_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'))
json.dump(frozen, open(os.path.join(HERE, 'frozen.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(dict(tune=tune, val=val, full=full), open(os.path.join(DEV, 'dev_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

rk = sorted(tune, key=lambda k: -tune[k]['Jbar'])


def fmt(v):
    return '–' if v is None else f'{v:.4f}'


tab = '\n'.join(f"| {k} | {CD.ROUND.get(k, '1, 3b')} | {json.dumps(CD.CFG.get(k), ensure_ascii=False) if k in CD.CFG else 'метод BSS'} | "
                f"{tune[k]['Jbar']:.4f} | {tune[k]['R']:.4f} | {fmt(val[k]['Jbar'] if k in val else None)} | "
                f"{fmt(full[k]['Jbar'] if k in full else None)} |" for k in rk)
P = f"""# Протокол підтверджувальної перевірки методу HBF (стаття 1)

Зафіксовано: {frozen['frozen_at']} — до запуску будь-яких прогонів HBF, HBF-fix, HEDAC*-ε та сімейств чутливості
на підтверджувальних картах 11–60. Контрольна сума цього файлу — у PROTOCOL_S1.sha256.

## 1. Розроблення (завершено)

Карти розроблення: 100–119 (раунди 1–4) і 120–159 (етап 2, лише кандидати); 4 розстановки на карту; 7 умов: S1, S2,
S3 з часткою постраждалих поза моделлю ρ ∈ {{0; 0,2; 0,4; 0,6; 0,8}}. Підтверджувальні карти 11–60 у розробленні
не використовувалися. Кількість оцінених конфігурацій: сімейство HBF — {n_hbf} (з них 4 — з розподілом ролей), абляції β = 0 — 2,
значення ε конкурента HEDAC*-ε — {n_eps}. Причину етапу 2 й правило викладено в dev/STAGE2_RULE.md.

Правило відбору: {frozen['rule']}; нічия — менше K, потім менша w_rest.

| Конфігурація | Раунд | Параметри | J̄ 100–119 | R 100–119 | J̄ 120–159 | J̄ 100–159 |
|---|---|---|---|---|---|---|
{tab}

## 2. Заморожені методи

* **HBF** = {hbf_star}: {json.dumps(cfg, ensure_ascii=False)}; μ = 0,15; поле й правило руху — як у HEDAC*.
* **HBF-fix** (абляція) = HBF з β = 0 (ваги гіпотез сталі, рівні апріорним).
* **HEDAC*-ε*** = HEDAC* з рівномірною складовою ε* = {eps} ({eps_star}).
* **HEDAC*** (ε = 0,05) і **BSS** — без змін; результати з архіву прогонів sar_swarm v2 (conf_main, mis_r00…r80).
* Сімейства чутливості (вторинний аналіз): HBF з апріорними налаштуваннями {', '.join(SENS_HBF)};
  HEDAC*-ε з ε ∈ {{{'; '.join(str(e).replace('.', ',') for e in SENS_EPS)}}}.

## 3. Дані підтвердження

* S1, S2: карти 11–60 × 4 розстановки (по 200 екземплярів).
* S3, серія ρ ∈ {{0; 0,2; 0,4; 0,6; 0,8}}: карти 11–40 × 4 (по 120 екземплярів).
* Додатково (поза первинною кінцевою точкою; лише HBF, HBF-fix, HEDAC*-ε*): S3 за замовчуванням (ρ = 0,4), карти 11–60 × 4.
* Зерна — як у пакеті (розстановка 1000m + v + o(s), сенсор crc32(інстанс), керування 7m + v); порівняння парні.
* Контроль відтворюваності: HEDAC* на S1–S3, карти 11–15 (60 прогонів) має побітово збігтися з архівом conf_main.

## 4. Гіпотези

* **H1** (первинна): D(HBF − HEDAC*-ε*) > 0.
* **H2**: D(HBF − HEDAC*) > 0.
* **H3**: D(HBF − BSS) > 0.
* **H4** (ціна за правильної карти): у кожній з умов S1, S2, S3 ρ = 0 HBF гірший за HEDAC* не більше ніж на δ = 0,02.
* **H5** (внесок навчання ваг): D(HBF − HBF-fix) > 0.
* **H6** (чутливість, описова): розкид J̄ (max − min) у сімействі апріорних налаштувань HBF (остаточне + 4 інші)
  менший, ніж у сімействі значень ε HEDAC*-ε; найгірше налаштування HBF має J̄ не нижчу за HEDAC* з ε = 0,05.

## 5. Кінцеві точки й тести

* D — середня за 7 умовами середня парна різниця SWS (рівні ваги умов). Одиниця незалежності — карта сценарію
  (S1: 50, S2: 50, S3: 30; внесок карти S3 — сума за п'ятьма ρ). Двобічний рандомізаційний тест знаків на рівні
  карт (100 000 перестановок, зерно 1); 95 % ДІ — стратифікований кластерний бутстреп карт (10 000, зерно 1).
* Поправка Холма в сім'ї H1, H2, H3, H5. Гіпотезу підтверджено, якщо p_Holm < 0,05 і D > 0. Якщо H1 не
  підтверджено, додатково повідомляється 95 % ДІ для D(HBF − HEDAC*-ε*) як межа можливої переваги чи відставання.
* H4: однобічний парний t-тест не-гіршості з межею −0,02 у кожній з трьох умов; підтверджено, якщо p < 0,05 у
  всіх трьох (принцип перетину-об'єднання, без поправки).
* Вторинні (описові, з поправкою Холма в межах порівняння): у кожній умові — парний Вілкоксон (zsplit), тест на
  рівні карт, A12, бутстреп 2000; відставання R від найкращого з оцінених статичних ε; чутливість D до
  τ ∈ {{120; 180; 240}} (перерахунок SWS з моментів виявлення); телеметрія ваг гіпотез.

## 6. Зобов'язання

Метод і ε* після цього моменту не змінюються. Усі результати звітуються незалежно від знаку. Будь-який аналіз,
не названий тут, позначається в статті як пост-фактум.
"""
open(os.path.join(HERE, 'PROTOCOL_S1.md'), 'w', encoding='utf-8').write(P)
h = hashlib.sha256(P.encode('utf-8')).hexdigest()
open(os.path.join(HERE, 'PROTOCOL_S1.sha256'), 'w').write(f'{h}  PROTOCOL_S1.md\n')
print('HBF*', hbf_star, cfg, 'J̄ 100–159', full[hbf_star]['Jbar'])
print('eps*', eps_star, eps, 'J̄ 100–159', full[eps_star]['Jbar'])
print('sens', SENS_HBF)
print('sha256', h)
