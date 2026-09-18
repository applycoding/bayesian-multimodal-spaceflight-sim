# Synthetic validation results — multimodal astronaut-health monitor

**Disclaimer:** All results below are from a **simulated** N=8 crew × T=90 day mission.
No real Artemis, ISS, or operational astronaut data were used. Numbers must not be
interpreted as clinical or flight-certified performance.

## Design (simulation)

- Latent state: fatigue, stress, recovery (VAR(1) Gaussian state-space).
- Induced fatigue ramps: crew [0, 2, 5], days 35–65.
- Environmental confound (high workload → activity/HRV shift without true fatigue):
  crew [1, 3, 4, 6], days 40–55.
- Channels: hrv, sleep_eff, cognitive, immune, activity, env_workload with heterogeneous missingness
  (cognitive ~every 3d, immune weekly).
- Elevated-fatigue day prevalence (GT latent > 1.0): 0.096.
- Seed: 42.

## Method comparison

| Method | F1 | AUROC | Precision | Recall | Mean delay (d) | False alert rate |
|--------|-----|-------|-----------|--------|----------------|------------------|
| Indep. thresholds | 0.256 | 0.773 | 0.149 | 0.899 | 0.00 | 0.544 |
| Univariate EWMA | 0.199 | 0.777 | 0.111 | 0.971 | 0.00 | 0.826 |
| **Proposed multimodal SSM** | **0.753** | **0.975** | **0.656** | **0.884** | **0.67** | **0.049** |
| Ablation: HRV-only SSM | 0.728 | 0.973 | 0.634 | 0.855 | 0.00 | 0.052 |
| Ablation: sleep-only SSM | 0.521 | 0.973 | 0.926 | 0.362 | 7.00 | 0.003 |

Detection delays for proposed (per ramp crew [0, 2, 5]): [1.0, 1.0, 0.0]

## Proposed model fidelity (vs simulated truth)

- Correlation(estimated fatigue, true fatigue): **0.879**
- RMSE(fatigue): **0.266**
- Calibrated fatigue alert threshold: 0.812; P(elevated)>=0.8

## Did the proposed method win?

- Beats baselines on F1: **True**
- Beats baselines on AUROC: **True**
- False alert rate (proposed): 0.049 vs thresholds 0.544, EWMA 0.826
- Note: baseline mean delays of 0.0 reflect near-constant alerting (very high FAR), not better detectors.

## Ablation takeaway

Multimodal F1 0.753 / AUROC 0.975 vs HRV-only F1 0.728 / AUROC 0.973 vs sleep-only F1 0.521 / AUROC 0.973. Multimodality mainly improves operating-point F1 / recall–precision balance versus sleep-only; AUROC is similar across SSM variants in this seed.

## Limitations (state plainly)

1. **Synthetic data only** — dynamics, loadings, noise, and missingness are authored by us;
   real physiological coupling and operational constraints differ.
2. Approximate inference (Kalman smoother + hierarchical shrinkage / ridge), not full MCMC.
3. Ground-truth labels use the simulated latent fatigue threshold (1.0);
   real “elevated fatigue” has no such oracle.
4. Small N (8 crew) and one random seed (42); results are illustrative, not a
   Monte Carlo uncertainty study.
5. Confound and ramp schedules are known by construction; operational deployment would
   face unlabeled change-points and distribution shift.

## Files

- Script: `simulate_and_evaluate.py`
- Metrics: `results/metrics.json`
- Figures: `figures/fig1_latent_vs_estimates.png`, `fig2_roc_comparison.png`,
  `fig3_delay_falsealarms.png`, `fig4_missingness_schematic.png`
