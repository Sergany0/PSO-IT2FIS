# PSO-IT2FIS: Delay-Dependent Intelligent Fusion for Loosely Coupled GPS/INS

Python reference implementation of the delay-dependent, PSO-tuned interval type-2
fuzzy inference system (PSO-IT2FIS) proposed in:

> A. M. Elsergany, M. A. Jaradat, and M. F. Abdel-Hafez, "Delay-Dependent Intelligent
> Fusion for Loosely Coupled GPS/INS During GPS Outages," *Soft Computing*, under review.

The results reported in the paper were obtained with the authors' MATLAB implementation;
this repository provides the algorithm of the proposed model as described in the paper.

## Method

For each position axis, an IT2FIS estimates the INS position error

    dP_INS(k) = IT2FIS( P_GPS(k), P_INS(k), P_INS(k-1), n(k) )
    P'_INS(k) = P_INS(k) + dP_INS(k)

where `n(k)` is the number of INS steps since the last GPS reading. During a GPS outage the
last GPS position is held and `n(k)` keeps accumulating, so a single online inference stage
covers both nominal operation and outages.

| Component | Paper | Module |
|---|---|---|
| Interval type-2 Gaussian MFs, 3 per input, 81-rule TSK grid | Eqs. (16)-(18) | `it2fis.py` |
| Firing intervals (product t-norm) | Eqs. (19)-(20) | `it2fis.py` |
| Centre-of-sets type reduction with EKM | Eqs. (21)-(24), Algorithms 1-2 | `it2fis.py` |
| PSO with adaptive inertia, RMSE fitness | Eqs. (25)-(27) | `pso.py` |
| Sequential tuning: consequents, upper MFs, lower MFs | Section III-B | `pso.py` |
| Delay input and position correction | Eqs. (13)-(15), Fig. 1 | `fusion.py` |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from pso_it2fis import IT2FIS, tune, tuning_data, correct

systems = []
for X, Y in tuning_data(p_ref, p_gps, p_ins, gps_available):   # one (X, Y) pair per axis
    fis = IT2FIS.grid_partition(X, Y, n_mf=3)
    tune(fis, X, Y, n_iter=30)
    systems.append(fis)

p_corrected = correct(systems, p_gps, p_ins, gps_available)
```

`p_ref`, `p_gps` and `p_ins` are `(K, 3)` position arrays and `gps_available` is a length-`K`
boolean array. `example.py` runs the full pipeline on synthetic signals.

## Data availability

The experimental datasets used in the paper are not included in this repository. They are
available from the corresponding author on reasonable request.

## License

See [LICENSE](LICENSE). Non-commercial academic research use is permitted with citation;
any other use requires prior written permission from the authors.

## Citation

If you use this code, please cite the paper above (see [CITATION.cff](CITATION.cff)).
