"""Interval type-2 TSK fuzzy inference system (paper Section III-A).

Each input universe is partitioned by N interval type-2 Gaussian membership
functions sharing a common centre c with an upper width sigma_u and a lower
width sigma_l = r * sigma_u, 0 < r <= 1 (Eqs. (17)-(18)). The rule base is the
full grid of M = N**n_inputs rules with constant consequents alpha_i (Eq. (16)).
Rule firing intervals use the product t-norm (Eqs. (19)-(20)), centre-of-sets
type reduction is solved with the enhanced Karnik-Mendel (EKM) algorithms
(Eqs. (21)-(23), Algorithms 1 and 2), and the crisp output is the mean of the
two end points (Eq. (24)).
"""
import itertools

import numpy as np


def gaussmf(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))


def ekm(alpha, f_lower, f_upper, side):
    """Enhanced Karnik-Mendel algorithm for one end point of Eq. (21).

    side='left' returns alpha_l (Algorithm 1), side='right' returns alpha_r
    (Algorithm 2).
    """
    order = np.argsort(alpha)
    a_s, fl, fu = alpha[order], f_lower[order], f_upper[order]
    m = len(a_s)
    if side == 'left':
        k = int(round(m / 2.4))
        w_first, w_last, sgn = fu, fl, 1.0
    else:
        k = int(round(m / 1.7))
        w_first, w_last, sgn = fl, fu, -1.0
    k = min(max(k, 1), m - 1)

    a = a_s[:k] @ w_first[:k] + a_s[k:] @ w_last[k:]
    b = w_first[:k].sum() + w_last[k:].sum()
    if b <= 0.0:
        return 0.0
    y = a / b
    for _ in range(m):
        k_new = int(np.clip(np.searchsorted(a_s, y, side='right'), 1, m - 1))
        if k_new == k:
            break
        s = np.sign(k_new - k)
        lo, hi = min(k, k_new), max(k, k_new)
        d = fu[lo:hi] - fl[lo:hi]
        a += sgn * s * (a_s[lo:hi] @ d)
        b += sgn * s * d.sum()
        k, y = k_new, a / b
    return y


class IT2FIS:
    """Interval type-2 TSK FIS with grid-partitioned Gaussian antecedents."""

    def __init__(self, centers, sigma_upper, lower_ratio, consequents):
        self.centers = np.asarray(centers, float)          # (n_inputs, N)
        self.sigma_upper = np.asarray(sigma_upper, float)  # (n_inputs, N)
        self.lower_ratio = np.asarray(lower_ratio, float)  # (n_inputs, N)
        self.consequents = np.asarray(consequents, float)  # (M,)
        n_inputs, n_mf = self.centers.shape
        self.rules = np.array(list(itertools.product(range(n_mf), repeat=n_inputs)))

    @classmethod
    def grid_partition(cls, X, Y, n_mf=3, lower_ratio=0.8):
        """Grid-partitioned initialisation from the tuning data (Section III-B)."""
        X = np.asarray(X, float)
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        centers = np.linspace(lo, lo + span, n_mf).T
        # adjacent membership functions intersect at a grade of 0.5
        sigma = span / (2.0 * (n_mf - 1) * np.sqrt(2.0 * np.log(2.0)))
        sigma_upper = np.repeat(sigma[:, None], n_mf, axis=1)
        ratio = np.full_like(sigma_upper, lower_ratio)
        consequents = np.full(n_mf ** X.shape[1], float(np.mean(Y)))
        return cls(centers, sigma_upper, ratio, consequents)

    def firing_intervals(self, X):
        """Lower and upper rule firing levels, Eqs. (19)-(20). X: (K, n_inputs)."""
        X = np.atleast_2d(np.asarray(X, float))
        n_inputs = self.centers.shape[0]
        x = X[:, :, None]
        mu_u = gaussmf(x, self.centers, self.sigma_upper)
        mu_l = gaussmf(x, self.centers, self.sigma_upper * self.lower_ratio)
        idx = np.arange(n_inputs)
        f_u = np.prod(mu_u[:, idx, self.rules], axis=2)
        f_l = np.prod(mu_l[:, idx, self.rules], axis=2)
        return f_l, f_u

    def infer(self, x):
        """Online inference for one sample with EKM type reduction, Eq. (24)."""
        f_l, f_u = self.firing_intervals(x)
        a_l = ekm(self.consequents, f_l[0], f_u[0], 'left')
        a_r = ekm(self.consequents, f_l[0], f_u[0], 'right')
        return 0.5 * (a_l + a_r)

    def predict(self, X):
        """Batch inference used during tuning.

        Evaluates every switch point of Eqs. (22)-(23) at once, which yields
        the same end points as the EKM iterations of infer().
        """
        f_l, f_u = self.firing_intervals(X)
        order = np.argsort(self.consequents)
        a_s, fl, fu = self.consequents[order], f_l[:, order], f_u[:, order]

        def end_point(w_first, w_last, pick):
            num_first = np.cumsum(a_s * w_first, axis=1)[:, :-1]
            den_first = np.cumsum(w_first, axis=1)[:, :-1]
            num_last = np.cumsum((a_s * w_last)[:, ::-1], axis=1)[:, ::-1][:, 1:]
            den_last = np.cumsum(w_last[:, ::-1], axis=1)[:, ::-1][:, 1:]
            den = den_first + den_last
            y = np.divide(num_first + num_last, den, out=np.zeros_like(den), where=den > 0)
            return pick(y, axis=1)

        a_l = end_point(fu, fl, np.min)
        a_r = end_point(fl, fu, np.max)
        return 0.5 * (a_l + a_r)
