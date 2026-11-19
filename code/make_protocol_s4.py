# -*- coding: utf-8 -*-
"""Механічний відбір остаточної конфігурації S4 за результатами розроблення (карти 100–139), запис frozen_s4.json,
конфігурацій підтвердження й протоколу PROTOCOL_S4.md (з контрольною сумою). Запускається ДО прогонів на картах 61–99."""
import json, datetime, hashlib, os
import numpy as np
import dev_sum_s4 as D
import make_dev_s4 as MD

R1, R2 = D.load(['s4_dev_r1']), D.load(['s4_dev_r2'])
cand = json.load(open('configs_s4/dev_r2.json', encoding='utf-8'))['methods']
dev = {}
for m in R2:
    per = [float(np.mean(R1[m][c] + R2[m][c])) for c in D.CONDS]
    dev[m] = dict(J=float(np.mean(per)), per=dict(zip(D.CONDS, per)), n_per_cond=len(R1[m]['A0']) + len(R2[m]['A0']))
hs = [m for m in dev if m.startswith('HS:')]
nhyp = lambda m: len(cand[m]['params']['spec'])
hs.sort(key=lambda m: (-round(dev[m]['J'], 9), nhyp(m), cand[m]['params']['w_rest']))
best = hs[0]
eps_c = [m for m in dev if m.startswith('HEDAC*')]
eps_c.sort(key=lambda m: -round(dev[m]['J'], 9))
eps_best = eps_c[0]
eps_star = 0.05 if eps_best == 'HEDAC*' else int(eps_best.split('|e')[1]) / 100
fz = dict(hbfs_name=best, hbfs=cand[best], eps_name=eps_best, eps_star=eps_star,
          dev=dev, dev_maps='100–119 (2 розстановки, раунд 1) + 120–139 (2 розстановки, етап 2)',
          rule='кандидати — 5 найкращих HBF-S і 3 найкращі ε за J̄ на 100–119; вибір — максимум J̄ на 100–139')
json.dump(fz, open('frozen_s4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

spec = cand[best]['params']
HBFS = {'cls': 'HBFS', 'params': dict(spec)}
FIX = {'cls': 'HBFS', 'params': dict(spec, beta=0.0)}
EPS = {'cls': 'HEDAC', 'row': eps_star} if eps_best != 'HEDAC*' else {'cls': 'HEDAC'}
main = {'HEDAC*': {'cls': 'HEDAC'}, 'HEDAC*-eps*': EPS, 'HBF-U': cand['HBF-U'], 'HBF-S': HBFS, 'HBF-S-fix': FIX,
        'BSS': {'cls': 'BSS'}, 'ORACLE': {'cls': 'ORACLE'}}
json.dump(dict(name='s4_conf_main', conds=['A0', 'A40', 'A80', 'B0', 'B40', 'B80'], maps=[61, 99], draws=4, methods=main,
               description='Стаття 1, підтвердження S4: карти 61–99, 4 розстановки'),
          open('configs_s4/conf_main.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
nv = {'HEDAC*': {'cls': 'HEDAC'}, 'HEDAC*|e20': {'cls': 'HEDAC', 'row': 0.2}, 'HBF': cand['HBF-U'],
      'HBF-fix': {'cls': 'HBF', 'params': {'rhos': [0.05, 0.3, 0.8], 'w_rest': 0.5, 'beta': 0.0}}}
json.dump(dict(name='s4_conf_nv', conds=['U40n3', 'U40n8', 'U80n3', 'U80n8'], maps=[61, 99], draws=4, methods=nv,
               description='Стаття 1, підтвердження: серія N_v (3 і 8), S3-геометрія, ρ = 0,4 і 0,8, карти 61–99'),
          open('configs_s4/conf_nv.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')
top = sorted(dev.items(), key=lambda kv: -kv[1]['J'])
tab = '\n'.join(f"| {m} | {v['J']:.4f} | " + ' | '.join(f"{v['per'][c]:.3f}" for c in D.CONDS) + ' |' for m, v in top)
P = f"""# Протокол підтверджувальних перевірок розширень статті 1 (S4, серія N_v, реальна місцевість)

Зафіксовано: {now} — до запуску будь-яких прогонів на картах 61–99. Контрольна сума цього файлу — у PROTOCOL_S4.sha256.

## 1. Розроблення (завершено)

Лише карти 100–139 (підтверджувальні карти 61–99 і карти статті 1 11–60 не використано). Раунд 1: карти 100–119 (2 розстановки),
10 конфігурацій HBF-S, 6 значень ε, HEDAC*, BSS, HBF-U (заморожений HBF статті 1), ORACLE-S. Етап 2 (правило —
dev/STAGE2_RULE_S4.md, зафіксовано до прогону): 5 найкращих HBF-S і 3 найкращі значення ε на картах 120–139.
Остаточний вибір — максимум J̄ (середнє за 6 умовами) на всіх картах розроблення 100–139 (80 екземплярів на умову).

| Метод | J̄ | A0 | A40 | A80 | B0 | B40 | B80 |
|---|---|---|---|---|---|---|---|
{tab}

Обрано: **{best}** (спец. гіпотез: {json.dumps(spec['spec'], ensure_ascii=False)}, w_rest = {spec['w_rest']}, β = {spec['beta']});
конкурент HEDAC*-ε*: **{eps_best}** (ε* = {eps_star}). Параметри — у frozen_s4.json.

## 2. Умови й дані підтвердження (S4)

A0, A40, A80 (зміщена ОВП, геометрія S1) і B0, B40, B80 (хибна категорія поведінки, геометрія S2); число — частка ρ
постраждалих поза моделлю, %. Карти 61–99 (39), 4 розстановки: 156 екземплярів на умову, 936 на метод.
Методи: HEDAC*, HEDAC*-ε*, HBF-U (заморожений HBF статті 1: ρ ∈ {{0,05; 0,3; 0,8}}, ваги 0,5/0,25/0,25, β = 1),
HBF-S, HBF-S-fix (β = 0), BSS, ORACLE-S (довідковий: знає справжню суміш; у гіпотезах не бере участі).

## 3. Гіпотези й рішення (S4)

Первинна кінцева точка D — середня за шістьма умовами середня парна різниця SWS; одиниця незалежності — карта
(групи: A — 39 карт, B — 39 карт); двобічний рандомізаційний тест знаків (100 000 перестановок, зерно 1); 95 % ДІ —
стратифікований кластерний бутстреп (10 000, зерно 1). Поправка Холма в сім’ї H1–H4. Підтверджено, якщо p_Holm < 0,05 і D > 0.

* H1 (первинна): HBF-S кращий за HEDAC*-ε*.
* H2: HBF-S кращий за HBF-U (структуровані гіпотези важливіші за рівномірні).
* H3: HBF-S кращий за HBF-S-fix (навчання ваг дає внесок).
* H4: HBF-S кращий за HEDAC* (ε = 0,05).
* H5 (ціна за правильної карти): у кожній з умов A0 і B0 HBF-S гірший за HEDAC* не більше ніж на δ = 0,02
  (однобічний парний t-тест не-гіршості; підтверджено, якщо p < 0,05 в обох умовах).
* H6 (описова): частка запасу до ORACLE-S, яку закрито: (J̄(HBF-S) − J̄(ε*)) / (J̄(ORACLE-S) − J̄(ε*)).
Додатково (описово, без поправки): різниці за умовами (Вілкоксон, A12), BSS, чутливість до τ ∈ {{120; 240}}.

## 4. Серія N_v (швидші свідчення; передбачення Твердження 3)

Умови U40n3, U40n8, U80n3, U80n8: геометрія S3, ρ = 0,4 і 0,8, число постраждалих N_v = 3 і 8; карти 61–99, 4 розстановки.
Методи: HEDAC*, HEDAC*-0,2 (ε* статті 1), HBF і HBF-fix (заморожені, стаття 1).
* H7: виграш навчання G = J(HBF) − J(HBF-fix) за N_v = 8 більший, ніж за N_v = 3; тест — парні різниці G8 − G3 за картами,
  усереднені за двома ρ; двобічний рандомізаційний тест знаків (100 000, зерно 1). Підтверджено, якщо p < 0,05 і різниця > 0.
* Описово: HBF − HEDAC*-0,2 і HBF − HEDAC* у кожній умові.

## 5. Реальна місцевість (заморожені методи статті 1)

Карта OSM «karpaty» (sha256 у meta.json: 6d830e4f…), сітка 50×50 (клітинка ≈ 100 м), 5 агентів, T = 300; сценарії S1, S2, S3 (ρ = 0,4)
з інцидентами 61–99 (різні ОВП) і 4 розстановками: 156 екземплярів на сценарій. Методи: HEDAC*, HEDAC*-0,2, BSS, HBF, HBF-fix
(параметри статті 1, без повторного налаштування). D — середнє за трьома сценаріями; одиниця — інцидент (сценарій, номер); сім’я Холма H9–H12:
H9 HBF > HEDAC*-0,2; H10 HBF > HEDAC*; H11 HBF > HBF-fix; H12 HBF > BSS.

## 6. Відхилення

Усі відхилення від цього протоколу (затримки, технічні збої, змінений код) буде описано в статті. Результати з іншим хешем коду
в межах одного експерименту не змішуються. Негативні результати публікуються нарівні з позитивними.
"""
open('PROTOCOL_S4.md', 'w', encoding='utf-8').write(P)
open('PROTOCOL_S4.sha256', 'w').write(hashlib.sha256(P.encode('utf-8')).hexdigest() + '  PROTOCOL_S4.md\n')
print('обрано', best, '| ε*', eps_star, '|', now)
print(open('PROTOCOL_S4.sha256').read())
