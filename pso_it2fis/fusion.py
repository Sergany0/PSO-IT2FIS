"""Delay-dependent GPS/INS fusion (paper Section III, Eqs. (13)-(15), Fig. 1).

One IT2FIS per position axis estimates the INS position error
dP_INS(k) = IT2FIS(P_GPS(k), P_INS(k), P_INS(k-1), n(k)), and the corrected
position is P'_INS(k) = P_INS(k) + dP_INS(k). The delay input n(k) is the number
of INS steps since the last GPS reading; during an outage the last GPS position
is held and n(k) keeps accumulating.
"""
import numpy as np


def delay_input(gps_available):
    gps_available = np.asarray(gps_available, bool)
    n = np.zeros(gps_available.size)
    for k in range(1, n.size):
        n[k] = 0 if gps_available[k] else n[k - 1] + 1
    return n


def hold_gps(p_gps, gps_available):
    """Hold the last available GPS position while the signal is absent."""
    p = np.array(p_gps, float)
    for k in range(1, len(p)):
        if not gps_available[k]:
            p[k] = p[k - 1]
    return p


def fis_inputs(p_gps, p_ins, gps_available):
    """Input matrices [P_GPS(k), P_INS(k), P_INS(k-1), n(k)] for k >= 1, per axis."""
    p_gps = hold_gps(p_gps, gps_available)
    p_ins = np.asarray(p_ins, float)
    n = delay_input(gps_available)
    return [np.column_stack([p_gps[1:, a], p_ins[1:, a], p_ins[:-1, a], n[1:]])
            for a in range(p_ins.shape[1])]


def tuning_data(p_ref, p_gps, p_ins, gps_available):
    """Per-axis tuning pairs (X, Y) with target Y = P_ref - P_INS."""
    X = fis_inputs(p_gps, p_ins, gps_available)
    err = np.asarray(p_ref, float)[1:] - np.asarray(p_ins, float)[1:]
    return [(X[a], err[:, a]) for a in range(err.shape[1])]


def correct(systems, p_gps, p_ins, gps_available):
    """Apply Eq. (13) sample by sample, as in online operation."""
    out = np.array(p_ins, float)
    X = fis_inputs(p_gps, p_ins, gps_available)
    for a, fis in enumerate(systems):
        for k in range(X[a].shape[0]):
            out[k + 1, a] += fis.infer(X[a][k])
    return out
