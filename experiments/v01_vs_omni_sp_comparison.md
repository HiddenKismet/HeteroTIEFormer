# V0.1 versus Omni at each starting point

Negative `Δ` means V0.1 has lower MAE than Omni.  The Panasonic, TJU, and
NASA rows use the clean seed-42 one-step checkpoints; Oxford and Stanford use
the previously completed full runs and are therefore a secondary comparison.

| Dataset | SP | Omni observed MAE | V0.1 observed MAE | Δ | Omni recursive MAE | V0.1 recursive MAE | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Panasonic | 300 | 0.003222 | 0.003398 | +5.5% | 0.170767 | 0.218653 | +28.0% |
|  | 400 | 0.003288 | 0.003464 | +5.4% | 0.136421 | 0.179450 | +31.5% |
|  | 500 | 0.003431 | 0.003622 | +5.5% | 0.129703 | 0.162457 | +25.3% |
| TJU | 300 | 0.000920 | 0.000885 | **-3.8%** | 0.058018 | 0.035075 | **-39.5%** |
|  | 450 | 0.000982 | 0.000951 | **-3.2%** | 0.031138 | 0.013212 | **-57.6%** |
|  | 600 | 0.001051 | 0.001014 | **-3.5%** | 0.018790 | 0.007878 | **-58.1%** |
| NASA | 100 | 0.006987 | 0.009507 | +36.1% | 0.158726 | 0.173179 | +9.1% |
|  | 120 | 0.007321 | 0.009873 | +34.9% | 0.134123 | 0.158040 | +17.8% |
| Oxford | 6400 | 0.008806 | 0.010292 | +16.9% | 0.006987 | 0.008770 | +25.5% |
|  | 7000 | 0.013623 | 0.012439 | **-8.7%** | 0.010338 | 0.010262 | **-0.7%** |
| Stanford | 949 | 0.001427 | 0.001324 | **-7.2%** | 0.005097 | 0.004695 | **-7.9%** |
|  | 1204 | 0.001627 | 0.001451 | **-10.8%** | 0.003563 | 0.002379 | **-33.2%** |

The paper screenshot remains the Omni reference for the main PANASONIC,
CALCE, and GOTION tables.  The local clean Omni values above are not used to
replace those paper references.

