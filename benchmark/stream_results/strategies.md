Run 2026-11-13T07:15:09Z · commit 7c5a065 · Darwin arm64, 8 CPUs · Python 3.14.6, NumPy 2.5.4

| strategy | accuracy_mean | accuracy_std | drift_retrains_seed42 |
|---|---|---|---|
| train once | 0.5478 | 0.0113 | 0 |
| buffered (500) | 0.7329 | 0.005734 | 0 |
| buffered + DDM (500) | 0.7523 | 0.005654 | 1 |
