# Causal soft-sensor benchmark — measured results

Original synthetic data only. No legacy industrial observations are redistributed. This is a controlled numerical experiment, not validation on a carbon-capture plant. Six coupled stages, rotating noisy observation, causal auxiliary proxy. 24 training runs, 8 validation, 8 independent test, 8 kinetic-shift test runs. Three GRU optimisation seeds; each split retains the same process realisations.

| Split | Method | h=1 RMSE | h=3 | h=6 | h=12 |
|---|---|---:|---:|---:|---:|
| test | persistence | 0.0348 | 0.0345 | 0.0345 | 0.0520 |
| test | physics | 0.0186 | 0.0196 | 0.0232 | 0.0379 |
| test | gru | 0.0185 | 0.0193 | 0.0231 | 0.0344 |
| test | independent_fusion | 0.0175 | 0.0185 | 0.0222 | 0.0342 |
| test | correlated_fusion | 0.0175 | 0.0185 | 0.0223 | 0.0339 |
| ood | persistence | 0.0461 | 0.0464 | 0.0467 | 0.0749 |
| ood | physics | 0.0276 | 0.0299 | 0.0347 | 0.0577 |
| ood | gru | 0.0250 | 0.0290 | 0.0338 | 0.0541 |
| ood | independent_fusion | 0.0251 | 0.0286 | 0.0332 | 0.0539 |
| ood | correlated_fusion | 0.0250 | 0.0285 | 0.0332 | 0.0536 |

## Coverage and negative comparisons
| Split | Horizon | Correlated fusion coverage | Mean RMSE change versus independent fusion |
|---|---:|---:|---:|
| test | 1 | 0.954 | +0.000004 |
| test | 3 | 0.955 | -0.000008 |
| test | 6 | 0.961 | +0.000049 |
| test | 12 | 0.956 | -0.000247 |
| ood | 1 | 0.831 | -0.000129 |
| ood | 3 | 0.808 | -0.000004 |
| ood | 6 | 0.869 | -0.000028 |
| ood | 12 | 0.858 | -0.000208 |

## Methods and validity
GRU is trained on future observed points with a masked MSE, never on latent states. Latent-state error is simulation-only secondary analysis. Normalisation uses training runs only. Checkpoints are selected by validation observed MSE. Forecasts assume constant current control; no future measurements, labels or controls enter features. Prefix-invariance is unit-tested.
Fusion estimates second moments of forecast errors on validation data. With physics error ep and neural error en, neural weight w=(E[ep²]-E[ep en])/(E[ep²]+E[en²]-2E[ep en]), clipped to [0,1]. The comparator sets the cross moment to zero. This avoids an unjustified independence assumption but may overfit eight validation runs.
Intervals use the validation 95th percentile of absolute observed residuals per horizon. They are empirical residual intervals, not a distribution-free guarantee under dependent windows or drift. Overlap and temporal autocorrelation invalidate naive row bootstrap; intervals for method differences resample whole test runs. Eight clusters give imprecise intervals; three optimisation seeds do not add independent test trajectories.
## Legacy audit and limitations
The earlier my_v2 pipeline indexes a kinetic prior at t+h. Whether that is leakage depends on when that prior was available; this study does not assume a proven violation. Full-run residual interpolation is suitable as an offline target construction only if never used as an online feature. Saved metric versions differ from the earlier README. None of those old metrics is treated as a result here.
Simulated dynamics favour a related mechanistic model; mismatch and OOD kinetics only partially address that bias. No real-plant transfer, transformer baseline, future-control oracle or industrial deployment is claimed. These extensions are NOT RUN. Units and sample times are dimensionless. Negative comparisons remain in metrics.json.
