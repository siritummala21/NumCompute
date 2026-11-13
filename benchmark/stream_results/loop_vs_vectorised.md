Run 2026-11-13T07:15:09Z · commit 7c5a065 · Darwin arm64, 8 CPUs · Python 3.14.6, NumPy 2.5.4

| operation | n | loop_best_s | vectorised_best_s | speedup | loop_mean_s | vectorised_mean_s |
|---|---|---|---|---|---|---|
| best split (one node) | 250 | 0.04648 | 0.0003133 | 148.4 | 0.04967 | 0.0003551 |
| best split (one node) | 500 | 0.09036 | 0.0005635 | 160.4 | 0.091 | 0.0006193 |
| best split (one node) | 1000 | 0.221 | 0.001136 | 194.5 | 0.2331 | 0.001158 |
| best split (one node) | 2000 | 0.5947 | 0.002311 | 257.4 | 0.5965 | 0.00236 |
| tree fit (depth 6) | 500 | 0.4165 | 0.005515 | 75.53 | 0.4177 | 0.005565 |
| tree fit (depth 6) | 1000 | 0.9798 | 0.009071 | 108 | 0.991 | 0.009125 |
| tree fit (depth 6) | 2000 | 2.473 | 0.01709 | 144.7 | 2.475 | 0.01719 |
| tree predict | 20000 | 0.1919 | 0.003514 | 54.62 | 0.1929 | 0.003557 |
| running mean/var (40 chunks) | 20000 | 0.0389 | 0.005047 | 7.708 | 0.03948 | 0.005108 |
| confusion matrix | 200000 | 0.07276 | 0.007003 | 10.39 | 0.07324 | 0.007065 |
| stream run, buffer 500 | 3000 | 2.066 | 0.08039 | 25.7 | 2.066 | 0.0809 |
