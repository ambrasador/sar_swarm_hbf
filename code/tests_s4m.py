import numpy as np, s4m_env as M
for k in 'MN':
    mp=M.S4MMap(k,170); a=mp.alt_true
    print(k, a.sum().round(6), min(float(np.abs(a-h).sum()) for h in mp.hyp_alts)>0.5, mp.alt_info)
    r=np.random.default_rng(1); v,f=mp.sample_victims4(r,0.8,3); print(v,f)
import s4_env as S; b=S.S4Map('A',170); mp=M.S4MMap('M',170); print('prior same', np.array_equal(b.prior,mp.prior), np.array_equal(b.hyp_alts[0],mp.hyp_alts[0]))
