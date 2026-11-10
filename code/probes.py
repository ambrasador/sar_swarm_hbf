# -*- coding: utf-8 -*-
"""
Телеметрія методів (лише читання внутрішнього стану — поведінка НЕ змінюється; перевіряє тест).

BSS і похідні: pi1(t) — апостеріорна вага запасної (рівномірної) моделі, n_s(t) — кількість
скептиків; t_first_skeptic — перший крок, коли n_s > 0 (-1 — не з'явився). Закриває позначку К34
розділу 2: гіпотеза про пізнє накопичення свідчень на користь запасної моделі в S3.
SIFD і похідні: кількість різних ділянок, на які розподілено агентів (середнє за місію).
"""
import numpy as np
import novel as NV

EVERY = 5          # крок прорідження часових рядів


def _target(m):
    """Метод, стан якого читаємо (для контролера зв'язку — репліка агента 0)."""
    if hasattr(m, 'rep_of'):
        return lambda: m.rep_of[0].m
    return lambda: m


class BSSProbe:
    def __init__(self, m):
        self.get = _target(m)
        self.pi1, self.ns = [], []

    def after_act(self, env):
        b = self.get()
        w = b.weights()
        self.pi1.append(float(w[1]))
        self.ns.append(int(np.asarray(b.skeptic).sum()))

    def fields(self):
        ns = np.array(self.ns)
        first = int(np.argmax(ns > 0)) if (ns > 0).any() else -1
        return dict(bss_pi1=[round(x, 5) for x in self.pi1[::EVERY]],
                    bss_ns=[int(x) for x in self.ns[::EVERY]],
                    bss_t_first_skeptic=first, bss_pi1_max=float(max(self.pi1) if self.pi1 else 0.0),
                    bss_ns_mean=float(ns.mean()) if len(ns) else 0.0)


class SIFDProbe:
    def __init__(self, m):
        self.get = _target(m)
        self.npatch = []

    def after_act(self, env):
        a = np.asarray(self.get().assign)
        self.npatch.append(len(set(a[a >= 0].tolist())))

    def fields(self):
        return dict(sifd_patches_mean=float(np.mean(self.npatch)) if self.npatch else 0.0)


def attach(m):
    """Повертає зонд для методу або None."""
    tgt = _target(m)()
    if isinstance(tgt, NV.BSS):
        return BSSProbe(m)
    if isinstance(tgt, NV.SIFD):
        return SIFDProbe(m)
    return None
