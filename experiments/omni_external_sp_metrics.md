# OmniTIEFormer per-start-point results

These are the locally trained Omni checkpoints for datasets outside the
paper screenshot.  `observed` refreshes the input with measured capacity at
each step; `recursive` feeds back the prediction.  MAE/RMSE are in the source
capacity units (Ah for Panasonic/TJU/NASA/Oxford; SOH units for Stanford).

## Trajectory metrics

| Dataset / run | SP | Observed MAE | Observed RMSE | Observed R² | Recursive MAE | Recursive RMSE | Recursive R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Panasonic / clean one-step | 300 | 0.003222 | 0.010313 | 0.997202 | 0.170767 | 0.191652 | 0.033917 |
|  | 400 | 0.003288 | 0.010716 | 0.995677 | 0.136421 | 0.157357 | 0.068011 |
|  | 500 | 0.003431 | 0.011533 | 0.991533 | 0.129703 | 0.149082 | -0.414757 |
| TJU / clean one-step | 300 | 0.000920 | 0.001499 | 0.999866 | 0.058018 | 0.064451 | 0.751839 |
|  | 450 | 0.000982 | 0.001616 | 0.999707 | 0.031138 | 0.036553 | 0.850163 |
|  | 600 | 0.001051 | 0.001731 | 0.999291 | 0.018790 | 0.024481 | 0.858167 |
| NASA / clean one-step | 100 | 0.006987 | 0.010344 | 0.969312 | 0.158726 | 0.183370 | -8.644489 |
|  | 120 | 0.007321 | 0.011273 | 0.913673 | 0.134123 | 0.157713 | -15.895539 |
| Oxford / previous full run | 6400 | 0.008806 | 0.017557 | -0.242057 | 0.006987 | 0.014127 | 0.195806 |
|  | 7000 | 0.013623 | 0.023094 | -0.534501 | 0.010338 | 0.018569 | 0.007897 |
| Stanford / previous full run | 949 | 0.001427 | 0.001938 | 0.993768 | 0.005097 | 0.006173 | 0.936802 |
|  | 1204 | 0.001627 | 0.002178 | 0.984164 | 0.003563 | 0.004298 | 0.938364 |

## RUL errors by SP

`AE/RE` are reported only when both the measured and predicted trajectories
cross the selected EOL threshold.  A dash means the crossing was undefined.

| Dataset / run | SP | Observed AE (cycle) | Observed RE | Recursive AE (cycle) | Recursive RE |
|---|---:|---:|---:|---:|---:|
| Panasonic / clean one-step | 300 | 0.483 | 0.001698 | 104.426 | 0.366900 |
|  | 400 | 0.483 | 0.002618 | 52.102 | 0.282218 |
|  | 500 | 0.483 | 0.005711 | 25.171 | 0.297466 |
| TJU / clean one-step | 300 | 0.824 | 0.001722 | — | — |
|  | 450 | 0.824 | 0.002509 | 74.050 | 0.225474 |
|  | 600 | 0.824 | 0.004618 | 36.573 | 0.204985 |
| NASA / clean one-step | 100 | 0.337 | 0.013881 | 11.917 | 0.491085 |
|  | 120 | 0.337 | 0.078937 | 3.290 | 0.770867 |
| Oxford / previous full run | 6400 | — | — | — | — |
|  | 7000 | — | — | — | — |
| Stanford / previous full run | 949 | 19.003 | 0.024361 | 88.010 | 0.112826 |
|  | 1204 | 19.003 | 0.036196 | 53.033 | 0.101016 |

The paper screenshot uses starting-point sets PANASONIC `{300, 400, 500}`,
CALCE `{300, 400, 500}`, and GOTION `{450, 600, 750}`.  Their Omni aggregate
values remain the fixed real-capacity-filling references in
`omni_paper_real_capacity_reference.md`; they are not replaced by these local
checkpoints.

