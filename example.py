"""Minimal usage example on synthetic signals (illustration only).

The signals below are generated here to show the API end to end; they are not
the experimental data of the paper.
"""
import numpy as np

from pso_it2fis import IT2FIS, correct, rmse, tune, tuning_data

rng = np.random.default_rng(1)
f_ins, f_gps, T = 50, 4, 60.0                     # Hz, Hz, s
t = np.arange(0, T, 1 / f_ins)
p_ref = np.column_stack([20 * np.sin(0.05 * t), 15 * np.cos(0.04 * t), 0.2 * np.sin(0.1 * t)])

gps = (np.arange(t.size) % round(f_ins / f_gps)) == 0
gps[(t > 30) & (t < 40)] = False                  # simulated 10 s outage
p_gps = p_ref + rng.normal(0, 0.5, p_ref.shape)

n_since = np.zeros(t.size)                        # INS drift grows between GPS fixes
for k in range(1, t.size):
    n_since[k] = 0 if gps[k] else n_since[k - 1] + 1
p_ins = p_ref + 1e-4 * n_since[:, None] ** 2 + rng.normal(0, 0.05, p_ref.shape)

systems = []
for X, Y in tuning_data(p_ref, p_gps, p_ins, gps):
    fis = IT2FIS.grid_partition(X, Y, n_mf=3)
    tune(fis, X, Y, n_iter=5)                     # the paper uses 30 tuning epochs
    systems.append(fis)

p_corr = correct(systems, p_gps, p_ins, gps)
for a, name in enumerate('xyz'):
    print(f'{name}: RMSE INS = {rmse(p_ref[:, a], p_ins[:, a]):.3f} m, '
          f'RMSE corrected = {rmse(p_ref[:, a], p_corr[:, a]):.3f} m')
