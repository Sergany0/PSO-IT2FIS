"""PSO tuning of the IT2FIS (paper Section III-B).

Particles follow Eqs. (25)-(26) with a swarm size of min(100, 10 * n_v),
c1 = c2 = 1.49 and an adaptive inertia weight bounded within [0.1, 1.1]. The
fitness is the RMSE of Eq. (27). Tuning is sequential: the rule consequents
first (antecedents fixed at their grid-partitioned values), then the upper
membership functions, then the lower membership functions.
"""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pso(fitness, lb, ub, n_iter=30, c1=1.49, c2=1.49, w_range=(0.1, 1.1), seed=0):
    """Minimise fitness(x) within the box [lb, ub]; returns (best_x, best_f)."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    n_v = lb.size
    n_p = min(100, 10 * n_v)
    rng = np.random.default_rng(seed)
    span = ub - lb

    x = rng.uniform(lb, ub, (n_p, n_v))
    v = rng.uniform(-span, span, (n_p, n_v))
    f = np.array([fitness(p) for p in x])
    p_best, f_best = x.copy(), f.copy()
    g = int(np.argmin(f_best))
    g_best, g_f = p_best[g].copy(), f_best[g]

    w, stall = w_range[1], 0
    for _ in range(n_iter):
        r1, r2 = rng.random((2, n_p, n_v))
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = np.clip(x + v, lb, ub)
        f = np.array([fitness(p) for p in x])
        better = f < f_best
        p_best[better], f_best[better] = x[better], f[better]

        g = int(np.argmin(f_best))
        if f_best[g] < g_f:
            g_best, g_f = p_best[g].copy(), f_best[g]
            stall = max(0, stall - 1)
        else:
            stall += 1
        # adaptive inertia weight
        if stall < 2:
            w = min(2.0 * w, w_range[1])
        elif stall > 5:
            w = max(0.5 * w, w_range[0])
    return g_best, g_f


def tune(fis, X, Y, n_iter=30, seed=0):
    """Sequential PSO tuning of an IT2FIS on the tuning data (X, Y), in place."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    lo, hi = X.min(axis=0), X.max(axis=0)

    # Stage 1: rule consequents.
    def f_cons(p):
        fis.consequents = p
        return rmse(Y, fis.predict(X))
    n_rules = fis.consequents.size
    best, _ = pso(f_cons, np.full(n_rules, Y.min()), np.full(n_rules, Y.max()), n_iter, seed=seed)
    fis.consequents = best

    # Stage 2: upper membership functions (centres and upper widths).
    shape = fis.centers.shape
    sigma0 = fis.sigma_upper.copy()

    def f_upper(p):
        fis.centers = p[:p.size // 2].reshape(shape)
        fis.sigma_upper = p[p.size // 2:].reshape(shape)
        return rmse(Y, fis.predict(X))
    c_lb, c_ub = np.repeat(lo[:, None], shape[1], 1), np.repeat(hi[:, None], shape[1], 1)
    lb = np.concatenate([c_lb.ravel(), sigma0.ravel()])
    ub = np.concatenate([c_ub.ravel(), 3.5 * sigma0.ravel()])
    best, _ = pso(f_upper, lb, ub, n_iter, seed=seed)
    f_upper(best)

    # Stage 3: lower membership functions (lower-to-upper width ratio).
    def f_lower(p):
        fis.lower_ratio = p.reshape(shape)
        return rmse(Y, fis.predict(X))
    best, err = pso(f_lower, np.full(fis.lower_ratio.size, 0.1),
                    np.ones(fis.lower_ratio.size), n_iter, seed=seed)
    f_lower(best)
    return fis, err
