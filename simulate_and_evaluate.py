#!/usr/bin/env python3
"""
Honest synthetic validation study for a hierarchical Bayesian-style
multimodal longitudinal astronaut-health monitor.

IMPORTANT: All data are SIMULATED. No real Artemis/ISS/crew data are used.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths & reproducibility
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

N_CREW = 8
T = 90
LATENT_DIM = 3  # fatigue, stress, recovery
CHANNEL_NAMES = [
    "hrv",
    "sleep_eff",
    "cognitive",
    "immune",
    "activity",
    "env_workload",
]
N_CH = len(CHANNEL_NAMES)

# Observation schedule / missingness base rates
# immune weekly, cognitive ~every 3 days; others daily with some missing
OBS_PROB = {
    "hrv": 0.92,
    "sleep_eff": 0.90,
    "cognitive": 0.38,  # ~every 3 days
    "immune": 0.16,  # weekly-ish
    "activity": 0.95,
    "env_workload": 0.98,
}

# Fatigue ramp: crew 0, 2, 5 get mid-mission fatigue increase
RAMP_CREW = [0, 2, 5]
RAMP_START = 35
RAMP_END = 65
RAMP_MAGNITUDE = 1.8

# Environmental confound: days 40-55 high workload -> activity/HRV shift
# without true fatigue for NON-ramp crew (and partially for ramp crew)
CONFOUND_START = 40
CONFOUND_END = 55
CONFOUND_CREW = [1, 3, 4, 6]  # crew without true fatigue ramp

FATIGUE_GT_THRESH = 1.0  # ground-truth elevated fatigue label threshold
ALERT_PROB_THRESH = 0.8
PREFLIGHT = slice(0, 7)


# ===========================================================================
# 1. Simulation
# ===========================================================================
def simulate_mission(rng: np.random.Generator):
    """Simulate latent states and multimodal observations for N_CREW x T."""
    # Shared group-level observation loadings (C matrix: channels x latent)
    # Columns: fatigue, stress, recovery
    C_group = np.array(
        [
            [-0.9, -0.3, 0.4],  # HRV: lower when fatigued/stressed
            [-0.7, -0.2, 0.6],  # sleep efficiency
            [-0.8, -0.5, 0.5],  # cognitive
            [-0.5, -0.4, 0.3],  # immune marker (lower = worse)
            [0.3, 0.6, -0.2],  # activity load (higher under stress)
            [0.1, 0.8, -0.1],  # env workload index
        ],
        dtype=float,
    )

    # Process model: mild VAR(1)
    A = np.array(
        [
            [0.85, 0.04, -0.10],
            [0.05, 0.82, -0.05],
            [-0.05, -0.06, 0.86],
        ]
    )
    Q = np.diag([0.035, 0.04, 0.03])  # process noise

    # Channel observation noise (R diagonal)
    R_diag = np.array([0.25, 0.30, 0.40, 0.55, 0.35, 0.20])

    true_x = np.zeros((N_CREW, T, LATENT_DIM))
    Y = np.full((N_CREW, T, N_CH), np.nan)
    mask = np.zeros((N_CREW, T, N_CH), dtype=bool)
    C_crew = np.zeros((N_CREW, N_CH, LATENT_DIM))

    for k in range(N_CREW):
        # Hierarchical shrinkage: crew loadings ~ N(C_group, small)
        C_k = C_group + rng.normal(0, 0.08, size=C_group.shape)
        C_crew[k] = C_k

        # Initial latent near zero (preflight baseline)
        x = rng.normal(0, 0.2, size=LATENT_DIM)
        true_x[k, 0] = x

        for t in range(T):
            if t > 0:
                x = A @ true_x[k, t - 1] + rng.multivariate_normal(
                    np.zeros(LATENT_DIM), Q
                )
            # Induced fatigue ramp (change-point): gradual rise then hold
            if k in RAMP_CREW and RAMP_START <= t <= RAMP_END:
                progress = (t - RAMP_START) / max(1, RAMP_END - RAMP_START)
                # Rise over first 2/3 of window, then hold near magnitude
                rise = min(1.0, progress / 0.67)
                target = RAMP_MAGNITUDE * rise
                x[0] = 0.70 * x[0] + 0.30 * target
                x[1] = 0.90 * x[1] + 0.10 * (0.6 * rise)
                x[2] = 0.90 * x[2] + 0.10 * (-0.5 * rise)

            true_x[k, t] = x

            # Environmental confound: affects activity & HRV observations only
            confound = 0.0
            if CONFOUND_START <= t <= CONFOUND_END:
                confound = 1.2 if k in CONFOUND_CREW else 0.4

            mu = C_k @ x
            # Apply confound to activity (idx 4) and HRV (idx 0) observation means
            mu[4] += confound * 0.9  # higher activity reading
            mu[0] -= confound * 0.5  # lower HRV (looks like fatigue) but latent ok

            y = mu + rng.normal(0, np.sqrt(R_diag))

            for c, name in enumerate(CHANNEL_NAMES):
                p = OBS_PROB[name]
                # Extra missingness during high-workload confound window
                if CONFOUND_START <= t <= CONFOUND_END and name in (
                    "cognitive",
                    "immune",
                ):
                    p *= 0.7
                if rng.random() < p:
                    Y[k, t, c] = y[c]
                    mask[k, t, c] = True

    meta = {
        "A": A,
        "Q": Q,
        "R_diag": R_diag,
        "C_group": C_group,
        "C_crew": C_crew,
        "ramp_crew": RAMP_CREW,
        "ramp_start": RAMP_START,
        "ramp_end": RAMP_END,
        "confound_crew": CONFOUND_CREW,
        "confound_start": CONFOUND_START,
        "confound_end": CONFOUND_END,
    }
    return true_x, Y, mask, meta


# ===========================================================================
# 2. Methods
# ===========================================================================
def zscore_from_preflight(Y, mask, preflight=PREFLIGHT):
    """Per-crew, per-channel z-scores using days 1-7 baseline."""
    Z = np.full_like(Y, np.nan)
    for k in range(Y.shape[0]):
        for c in range(Y.shape[2]):
            base = Y[k, preflight, c][mask[k, preflight, c]]
            if base.size < 2:
                mu, sd = 0.0, 1.0
            else:
                mu, sd = float(np.nanmean(base)), float(np.nanstd(base))
                if sd < 1e-6:
                    sd = 1.0
            obs = mask[k, :, c]
            Z[k, obs, c] = (Y[k, obs, c] - mu) / sd
    return Z


def method_independent_thresholds(Z, mask, thresh=2.0):
    """Baseline A: flag if any observed channel |z| > thresh."""
    alerts = np.zeros((N_CREW, T), dtype=bool)
    scores = np.zeros((N_CREW, T))
    for k in range(N_CREW):
        for t in range(T):
            vals = Z[k, t, mask[k, t]]
            if vals.size == 0:
                scores[k, t] = 0.0
                continue
            mx = float(np.nanmax(np.abs(vals)))
            scores[k, t] = mx
            alerts[k, t] = mx > thresh
    return alerts, scores


def ewma_channel(z, lam=0.2):
    """Univariate EWMA on partially observed z series; NaN days held."""
    out = np.full_like(z, np.nan, dtype=float)
    s = 0.0
    started = False
    for t in range(len(z)):
        if np.isnan(z[t]):
            if started:
                out[t] = s
            continue
        if not started:
            s = z[t]
            started = True
        else:
            s = lam * z[t] + (1 - lam) * s
        out[t] = s
    return out


def method_ewma(Z, mask, lam=0.2, L=2.5):
    """Baseline B: EWMA per channel, union of control-limit crossings."""
    alerts = np.zeros((N_CREW, T), dtype=bool)
    scores = np.zeros((N_CREW, T))
    # asymptotic EWMA variance factor for z~N(0,1): lam/(2-lam)
    sigma_e = np.sqrt(lam / (2 - lam))
    limit = L * sigma_e
    for k in range(N_CREW):
        ch_alert = np.zeros(T, dtype=bool)
        ch_score = np.zeros(T)
        for c in range(N_CH):
            z = np.where(mask[k, :, c], Z[k, :, c], np.nan)
            e = ewma_channel(z, lam=lam)
            a = np.abs(e) > limit
            a = np.nan_to_num(a.astype(float), nan=0.0).astype(bool)
            ch_alert |= a
            sc = np.nan_to_num(np.abs(e) / max(limit, 1e-6), nan=0.0)
            ch_score = np.maximum(ch_score, sc)
        alerts[k] = ch_alert
        scores[k] = ch_score
    return alerts, scores


def kalman_filter_smoother(y, mask_t, A, C, Q, R_diag, x0=None, P0=None):
    """
    Linear Gaussian SSM with missing observations (mask_t: T x n_ch bool).
    Returns filtered & RTS-smoothed means/covs for latent x.
    """
    Tloc, n_ch = y.shape
    d = A.shape[0]
    if x0 is None:
        x0 = np.zeros(d)
    if P0 is None:
        P0 = np.eye(d) * 0.5

    xf = np.zeros((Tloc, d))
    Pf = np.zeros((Tloc, d, d))
    xp = np.zeros((Tloc, d))
    Pp = np.zeros((Tloc, d, d))

    x_prev, P_prev = x0.copy(), P0.copy()
    for t in range(Tloc):
        # Predict
        x_pred = A @ x_prev
        P_pred = A @ P_prev @ A.T + Q
        xp[t], Pp[t] = x_pred, P_pred

        obs_idx = np.where(mask_t[t])[0]
        if obs_idx.size == 0:
            xf[t], Pf[t] = x_pred, P_pred
            x_prev, P_prev = x_pred, P_pred
            continue

        C_t = C[obs_idx]
        R_t = np.diag(R_diag[obs_idx])
        y_t = y[t, obs_idx]
        S = C_t @ P_pred @ C_t.T + R_t
        # Stabilize
        S = S + np.eye(S.shape[0]) * 1e-8
        K = P_pred @ C_t.T @ np.linalg.inv(S)
        innov = y_t - C_t @ x_pred
        x_upd = x_pred + K @ innov
        P_upd = (np.eye(d) - K @ C_t) @ P_pred
        # Symmetrize
        P_upd = 0.5 * (P_upd + P_upd.T)
        xf[t], Pf[t] = x_upd, P_upd
        x_prev, P_prev = x_upd, P_upd

    # RTS smoother
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(Tloc - 2, -1, -1):
        P_pred = Pp[t + 1]
        # G = Pf[t] A' inv(P_pred)
        try:
            G = Pf[t] @ A.T @ np.linalg.inv(P_pred + np.eye(d) * 1e-8)
        except np.linalg.LinAlgError:
            G = Pf[t] @ A.T @ np.linalg.pinv(P_pred)
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + G @ (Ps[t + 1] - P_pred) @ G.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)

    return xs, Ps, xf, Pf


def estimate_C_hierarchical(Y, mask, true_x_for_init=None, n_iter=8):
    """
    Approximate hierarchical loading estimation:
    1) Initialize C_k from OLS on a rough latent proxy (PCA of available channels)
    2) Shrink toward group mean (empirical Bayes / ridge toward mean)

    For honesty: we do NOT peek at true_x. Use PCA proxy for initial latent,
    then alternate Kalman + OLS with shrinkage.
    """
    # Build crew-wise PCA proxy for fatigue-ish first component
    from sklearn.decomposition import PCA

    C_est = np.zeros((N_CREW, N_CH, LATENT_DIM))
    # Initial A, Q, R from reasonable defaults (slightly misspecified vs truth)
    A = np.array(
        [
            [0.90, 0.04, -0.06],
            [0.05, 0.86, -0.04],
            [-0.03, -0.05, 0.88],
        ]
    )
    Q = np.diag([0.06, 0.06, 0.04])
    R_diag = np.array([0.30, 0.35, 0.45, 0.60, 0.40, 0.25])

    # Seed loadings from known physiological direction (weak prior knowledge)
    # plus data-driven scale — honest: use group prior mean as initialization
    C0 = np.array(
        [
            [-0.8, -0.25, 0.35],
            [-0.65, -0.2, 0.55],
            [-0.7, -0.45, 0.45],
            [-0.45, -0.35, 0.25],
            [0.25, 0.55, -0.15],
            [0.1, 0.7, -0.1],
        ],
        dtype=float,
    )

    for k in range(N_CREW):
        C_est[k] = C0 + RNG.normal(0, 0.02, size=C0.shape)

    # Alternating: smooth with current C, then OLS update with shrinkage
    xs_all = np.zeros((N_CREW, T, LATENT_DIM))
    Ps_all = np.zeros((N_CREW, T, LATENT_DIM, LATENT_DIM))

    for it in range(n_iter):
        for k in range(N_CREW):
            xs, Ps, _, _ = kalman_filter_smoother(
                Y[k], mask[k], A, C_est[k], Q, R_diag
            )
            xs_all[k] = xs
            Ps_all[k] = Ps

            # OLS per channel on smoothed latent (only observed days)
            C_new = C_est[k].copy()
            for c in range(N_CH):
                obs = mask[k, :, c]
                if obs.sum() < 10:
                    continue
                Xdes = xs[obs]  # Tobs x 3
                y = Y[k, obs, c]
                # Ridge OLS
                lam = 1.0
                XtX = Xdes.T @ Xdes + lam * np.eye(LATENT_DIM)
                Xty = Xdes.T @ y
                try:
                    C_new[c] = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    C_new[c] = np.linalg.lstsq(Xdes, y, rcond=None)[0]
            C_est[k] = C_new

        # Hierarchical shrinkage toward group mean
        C_mean = C_est.mean(axis=0)
        shrink = 0.4  # pull 40% toward group mean
        for k in range(N_CREW):
            C_est[k] = (1 - shrink) * C_est[k] + shrink * C_mean

    # Final smooth
    for k in range(N_CREW):
        xs, Ps, _, _ = kalman_filter_smoother(
            Y[k], mask[k], A, C_est[k], Q, R_diag
        )
        xs_all[k] = xs
        Ps_all[k] = Ps

    return xs_all, Ps_all, C_est, A, Q, R_diag


def method_proposed(Y, mask, fatigue_thresh=None, calib_scores=None):
    """
    Multimodal latent-state tracker with hierarchical loading shrinkage.
    Alert when P(fatigue > threshold | data) > 0.8 OR mean > calibrated cutoff.
    """
    xs, Ps, C_est, A, Q, R_diag = estimate_C_hierarchical(Y, mask)

    fatigue_mean = xs[:, :, 0]
    fatigue_var = Ps[:, :, 0, 0]
    fatigue_sd = np.sqrt(np.maximum(fatigue_var, 1e-8))

    # Calibrate threshold on preflight: set so that ~5% of preflight days alert
    # Use posterior mean cutoff and probability threshold jointly
    if fatigue_thresh is None:
        pre = fatigue_mean[:, PREFLIGHT].ravel()
        fatigue_thresh = float(np.nanpercentile(pre, 95) + 0.5)
        # Ensure reasonable range
        fatigue_thresh = max(0.6, min(fatigue_thresh, 1.5))

    # P(fatigue > thresh) under Gaussian posterior
    from scipy.stats import norm

    p_elev = 1.0 - norm.cdf(fatigue_thresh, loc=fatigue_mean, scale=fatigue_sd)

    alerts = (p_elev > ALERT_PROB_THRESH) | (fatigue_mean > fatigue_thresh)
    scores = p_elev.copy()  # continuous score for AUROC

    return alerts, scores, xs, Ps, fatigue_thresh, C_est


def method_statespace_subset(Y, mask, channel_indices, label="subset"):
    """Ablation: state-space using only selected channels."""
    Y_sub = Y[:, :, channel_indices]
    mask_sub = mask[:, :, channel_indices]
    n_ch = len(channel_indices)

    A = np.array(
        [
            [0.90, 0.04, -0.06],
            [0.05, 0.86, -0.04],
            [-0.03, -0.05, 0.88],
        ]
    )
    Q = np.diag([0.06, 0.06, 0.04])

    # Subset of default C0 and R
    C0_full = np.array(
        [
            [-0.8, -0.25, 0.35],
            [-0.65, -0.2, 0.55],
            [-0.7, -0.45, 0.45],
            [-0.45, -0.35, 0.25],
            [0.25, 0.55, -0.15],
            [0.1, 0.7, -0.1],
        ],
        dtype=float,
    )
    R_full = np.array([0.30, 0.35, 0.45, 0.60, 0.40, 0.25])
    C0 = C0_full[channel_indices]
    R_diag = R_full[channel_indices]

    xs_all = np.zeros((N_CREW, T, LATENT_DIM))
    Ps_all = np.zeros((N_CREW, T, LATENT_DIM, LATENT_DIM))
    C_est = np.zeros((N_CREW, n_ch, LATENT_DIM))
    for k in range(N_CREW):
        C_est[k] = C0.copy()

    for it in range(6):
        for k in range(N_CREW):
            xs, Ps, _, _ = kalman_filter_smoother(
                Y_sub[k], mask_sub[k], A, C_est[k], Q, R_diag
            )
            xs_all[k] = xs
            Ps_all[k] = Ps
            C_new = C_est[k].copy()
            for ci in range(n_ch):
                obs = mask_sub[k, :, ci]
                if obs.sum() < 10:
                    continue
                Xdes = xs[obs]
                y = Y_sub[k, obs, ci]
                lam = 1.0
                XtX = Xdes.T @ Xdes + lam * np.eye(LATENT_DIM)
                try:
                    C_new[ci] = np.linalg.solve(XtX, Xdes.T @ y)
                except np.linalg.LinAlgError:
                    C_new[ci] = np.linalg.lstsq(Xdes, y, rcond=None)[0]
            C_est[k] = C_new
        C_mean = C_est.mean(axis=0)
        for k in range(N_CREW):
            C_est[k] = 0.6 * C_est[k] + 0.4 * C_mean

    for k in range(N_CREW):
        xs, Ps, _, _ = kalman_filter_smoother(
            Y_sub[k], mask_sub[k], A, C_est[k], Q, R_diag
        )
        xs_all[k] = xs
        Ps_all[k] = Ps

    fatigue_mean = xs_all[:, :, 0]
    fatigue_sd = np.sqrt(np.maximum(Ps_all[:, :, 0, 0], 1e-8))
    pre = fatigue_mean[:, PREFLIGHT].ravel()
    fatigue_thresh = float(np.nanpercentile(pre, 95) + 0.5)
    fatigue_thresh = max(0.6, min(fatigue_thresh, 1.5))
    from scipy.stats import norm

    p_elev = 1.0 - norm.cdf(fatigue_thresh, loc=fatigue_mean, scale=fatigue_sd)
    alerts = (p_elev > ALERT_PROB_THRESH) | (fatigue_mean > fatigue_thresh)
    return alerts, p_elev, xs_all, Ps_all


# ===========================================================================
# 3. Metrics
# ===========================================================================
def ground_truth_labels(true_x, thresh=FATIGUE_GT_THRESH):
    """Day-level elevated fatigue from true latent state."""
    return true_x[:, :, 0] > thresh


def detection_delay(alerts, true_x, ramp_crew, ramp_start, ramp_end, gt_thresh):
    """
    For each ramp crew, delay = first alert day in [ramp_start, ramp_end+10]
    after true fatigue first exceeds gt_thresh, minus that onset day.
    Returns list of delays (days); inf if never detected.
    """
    delays = []
    for k in ramp_crew:
        fat = true_x[k, :, 0]
        onset_candidates = np.where(
            (np.arange(T) >= ramp_start) & (fat > gt_thresh)
        )[0]
        if onset_candidates.size == 0:
            # use ramp_start as nominal onset
            onset = ramp_start
        else:
            onset = int(onset_candidates[0])
        search = np.where(
            (np.arange(T) >= onset) & (np.arange(T) <= min(T - 1, ramp_end + 15))
        )[0]
        hit = None
        for t in search:
            if alerts[k, t]:
                hit = t
                break
        if hit is None:
            delays.append(np.nan)
        else:
            delays.append(float(hit - onset))
    return delays


def false_alert_rate(alerts, true_labels):
    """
    Fraction of non-event days (true_labels==False) that are alerted.
    """
    nonevent = ~true_labels
    if nonevent.sum() == 0:
        return 0.0
    return float(alerts[nonevent].mean())


def classification_metrics(alerts, scores, true_labels):
    y_true = true_labels.ravel().astype(int)
    y_pred = alerts.ravel().astype(int)
    y_score = scores.ravel().astype(float)
    # Handle constant scores
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1 and np.nanstd(y_score) > 1e-12:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def summarize_method(name, alerts, scores, true_x, true_labels, meta):
    delays = detection_delay(
        alerts,
        true_x,
        meta["ramp_crew"],
        meta["ramp_start"],
        meta["ramp_end"],
        FATIGUE_GT_THRESH,
    )
    far = false_alert_rate(alerts, true_labels)
    clf = classification_metrics(alerts, scores, true_labels)
    delay_clean = [d for d in delays if not (isinstance(d, float) and np.isnan(d))]
    return {
        "method": name,
        "detection_delay_days": delays,
        "mean_detection_delay": float(np.mean(delay_clean)) if delay_clean else None,
        "median_detection_delay": float(np.median(delay_clean))
        if delay_clean
        else None,
        "n_detected_ramps": len(delay_clean),
        "n_ramps": len(delays),
        "false_alert_rate": far,
        **clf,
        "alert_rate_overall": float(alerts.mean()),
    }


# ===========================================================================
# 4. Figures
# ===========================================================================
def fig1_latent_vs_estimates(true_x, xs, Ps, example_crew=(0, 1)):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    days = np.arange(1, T + 1)
    for ax, k in zip(axes, example_crew):
        true_f = true_x[k, :, 0]
        est = xs[k, :, 0]
        sd = np.sqrt(np.maximum(Ps[k, :, 0, 0], 1e-8))
        ax.plot(days, true_f, "k-", lw=2, label="True fatigue")
        ax.plot(days, est, "C0-", lw=1.5, label="Estimated (posterior mean)")
        ax.fill_between(
            days, est - 1.96 * sd, est + 1.96 * sd, color="C0", alpha=0.25, label="95% band"
        )
        if k in RAMP_CREW:
            ax.axvspan(RAMP_START + 1, RAMP_END + 1, color="red", alpha=0.12, label="Fatigue ramp")
        if k in CONFOUND_CREW:
            ax.axvspan(
                CONFOUND_START + 1,
                CONFOUND_END + 1,
                color="orange",
                alpha=0.12,
                label="Env confound",
            )
        else:
            ax.axvspan(
                CONFOUND_START + 1,
                CONFOUND_END + 1,
                color="orange",
                alpha=0.08,
            )
        ax.axhline(FATIGUE_GT_THRESH, color="gray", ls="--", lw=1, label="GT threshold")
        ax.set_ylabel("Fatigue latent")
        ax.set_title(f"Crew {k} ({'ramp' if k in RAMP_CREW else 'no ramp'})")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Mission day")
    fig.suptitle(
        "Synthetic study: true vs estimated fatigue (multimodal SSM)",
        fontsize=12,
    )
    fig.tight_layout()
    path = FIGURES / "fig1_latent_vs_estimates.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_roc(methods_scores_labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, labels in methods_scores_labels:
        y_true = labels.ravel().astype(int)
        y_score = scores.ravel().astype(float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: day-level elevated fatigue (synthetic)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "fig2_roc_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_delay_falsealarms(summaries):
    names = [s["method"] for s in summaries]
    delays = [
        s["mean_detection_delay"] if s["mean_detection_delay"] is not None else 0
        for s in summaries
    ]
    fars = [s["false_alert_rate"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    axes[0].bar(x, delays, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean detection delay (days)")
    axes[0].set_title("Detection delay (fatigue ramps)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, fars, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("False alert rate")
    axes[1].set_title("False alerts on non-event days")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Operating characteristics (synthetic validation)", fontsize=12)
    fig.tight_layout()
    path = FIGURES / "fig3_delay_falsealarms.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_missingness(mask, crew=0):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    data = mask[crew].T.astype(float)  # channels x days
    im = ax.imshow(data, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(N_CH))
    ax.set_yticklabels(CHANNEL_NAMES)
    ax.set_xlabel("Mission day (0-indexed)")
    ax.set_title(f"Data availability heatmap — Crew {crew} (white=missing, black=observed)")
    ax.axvline(CONFOUND_START, color="C1", ls="--", lw=1, label="Confound window")
    ax.axvline(CONFOUND_END, color="C1", ls="--", lw=1)
    if crew in RAMP_CREW:
        ax.axvline(RAMP_START, color="C3", ls=":", lw=1, label="Ramp window")
        ax.axvline(RAMP_END, color="C3", ls=":", lw=1)
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Observed")
    fig.tight_layout()
    path = FIGURES / "fig4_missingness_schematic.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# 5. Main
# ===========================================================================
def main():
    print("=" * 70)
    print("Synthetic astronaut-health monitor validation (NOT real Artemis/ISS data)")
    print("=" * 70)
    np.random.seed(SEED)

    true_x, Y, mask, meta = simulate_mission(RNG)
    true_labels = ground_truth_labels(true_x)
    print(f"Elevated-fatigue day prevalence: {true_labels.mean():.3f}")
    print(f"Observation rates: "
          + ", ".join(
              f"{n}={mask[:,:,i].mean():.2f}" for i, n in enumerate(CHANNEL_NAMES)
          ))

    Z = zscore_from_preflight(Y, mask)

    # --- Methods ---
    alerts_A, scores_A = method_independent_thresholds(Z, mask)
    alerts_B, scores_B = method_ewma(Z, mask)
    alerts_P, scores_P, xs_P, Ps_P, f_thresh, C_est = method_proposed(Y, mask)

    # Ablations
    hrv_idx = [CHANNEL_NAMES.index("hrv")]
    sleep_idx = [CHANNEL_NAMES.index("sleep_eff")]
    alerts_hrv, scores_hrv, xs_hrv, _ = method_statespace_subset(Y, mask, hrv_idx, "HRV-only")
    alerts_sleep, scores_sleep, xs_sleep, _ = method_statespace_subset(
        Y, mask, sleep_idx, "sleep-only"
    )

    summaries = [
        summarize_method("Indep. thresholds", alerts_A, scores_A, true_x, true_labels, meta),
        summarize_method("Univariate EWMA", alerts_B, scores_B, true_x, true_labels, meta),
        summarize_method("Proposed multimodal SSM", alerts_P, scores_P, true_x, true_labels, meta),
        summarize_method("Ablation: HRV-only SSM", alerts_hrv, scores_hrv, true_x, true_labels, meta),
        summarize_method("Ablation: sleep-only SSM", alerts_sleep, scores_sleep, true_x, true_labels, meta),
    ]

    # Correlation of estimated vs true fatigue (honesty check)
    corr_all = float(
        np.corrcoef(true_x[:, :, 0].ravel(), xs_P[:, :, 0].ravel())[0, 1]
    )
    rmse_fat = float(
        np.sqrt(np.mean((true_x[:, :, 0] - xs_P[:, :, 0]) ** 2))
    )

    # Pack metrics.json
    metrics = {
        "study_type": "synthetic_simulation",
        "disclaimer": (
            "All data are simulated. No real Artemis, ISS, or astronaut "
            "health records were used."
        ),
        "seed": SEED,
        "N_crew": N_CREW,
        "T_days": T,
        "latent_dim": LATENT_DIM,
        "channels": CHANNEL_NAMES,
        "observation_rates": {
            n: float(mask[:, :, i].mean()) for i, n in enumerate(CHANNEL_NAMES)
        },
        "elevated_fatigue_prevalence": float(true_labels.mean()),
        "gt_fatigue_threshold": FATIGUE_GT_THRESH,
        "proposed_fatigue_alert_threshold": float(f_thresh),
        "alert_probability_threshold": ALERT_PROB_THRESH,
        "ramp_crew": RAMP_CREW,
        "ramp_window_days": [RAMP_START, RAMP_END],
        "confound_crew": CONFOUND_CREW,
        "confound_window_days": [CONFOUND_START, CONFOUND_END],
        "proposed_fatigue_corr_with_truth": corr_all,
        "proposed_fatigue_rmse": rmse_fat,
        "methods": summaries,
    }

    # Convenience flat table
    metrics["comparison_table"] = {
        s["method"]: {
            "F1": s["f1"],
            "AUROC": s["auroc"],
            "precision": s["precision"],
            "recall": s["recall"],
            "mean_detection_delay": s["mean_detection_delay"],
            "false_alert_rate": s["false_alert_rate"],
            "n_detected_ramps": s["n_detected_ramps"],
        }
        for s in summaries
    }

    metrics_path = RESULTS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}")

    # Figures
    p1 = fig1_latent_vs_estimates(true_x, xs_P, Ps_P, example_crew=(0, 1))
    p2 = fig2_roc(
        [
            ("Indep. thresholds", scores_A, true_labels),
            ("Univariate EWMA", scores_B, true_labels),
            ("Proposed multimodal", scores_P, true_labels),
            ("HRV-only SSM", scores_hrv, true_labels),
            ("Sleep-only SSM", scores_sleep, true_labels),
        ]
    )
    p3 = fig3_delay_falsealarms(summaries)
    p4 = fig4_missingness(mask, crew=0)
    print(f"Figures: {p1}, {p2}, {p3}, {p4}")

    # summary.md
    prop = summaries[2]
    baseA = summaries[0]
    baseB = summaries[1]
    hrv_s = summaries[3]
    sleep_s = summaries[4]

    beats_f1 = prop["f1"] > max(baseA["f1"], baseB["f1"])
    beats_auc = prop["auroc"] > max(baseA["auroc"], baseB["auroc"])
    delay_ok = (
        prop["mean_detection_delay"] is not None
        and (
            baseA["mean_detection_delay"] is None
            or prop["mean_detection_delay"] <= baseA["mean_detection_delay"] + 2
        )
    )

    summary_md = f"""# Synthetic validation results — multimodal astronaut-health monitor

**Disclaimer:** All results below are from a **simulated** N={N_CREW} crew × T={T} day mission.
No real Artemis, ISS, or operational astronaut data were used. Numbers must not be
interpreted as clinical or flight-certified performance.

## Design (simulation)

- Latent state: fatigue, stress, recovery (VAR(1) Gaussian state-space).
- Induced fatigue ramps: crew {RAMP_CREW}, days {RAMP_START}–{RAMP_END}.
- Environmental confound (high workload → activity/HRV shift without true fatigue):
  crew {CONFOUND_CREW}, days {CONFOUND_START}–{CONFOUND_END}.
- Channels: {", ".join(CHANNEL_NAMES)} with heterogeneous missingness
  (cognitive ~every 3d, immune weekly).
- Seed: {SEED}.

## Method comparison

| Method | F1 | AUROC | Precision | Recall | Mean delay (d) | False alert rate |
|--------|-----|-------|-----------|--------|----------------|------------------|
| Indep. thresholds | {baseA['f1']:.3f} | {baseA['auroc']:.3f} | {baseA['precision']:.3f} | {baseA['recall']:.3f} | {baseA['mean_detection_delay']} | {baseA['false_alert_rate']:.3f} |
| Univariate EWMA | {baseB['f1']:.3f} | {baseB['auroc']:.3f} | {baseB['precision']:.3f} | {baseB['recall']:.3f} | {baseB['mean_detection_delay']} | {baseB['false_alert_rate']:.3f} |
| **Proposed multimodal SSM** | **{prop['f1']:.3f}** | **{prop['auroc']:.3f}** | **{prop['precision']:.3f}** | **{prop['recall']:.3f}** | **{prop['mean_detection_delay']}** | **{prop['false_alert_rate']:.3f}** |
| Ablation: HRV-only SSM | {hrv_s['f1']:.3f} | {hrv_s['auroc']:.3f} | {hrv_s['precision']:.3f} | {hrv_s['recall']:.3f} | {hrv_s['mean_detection_delay']} | {hrv_s['false_alert_rate']:.3f} |
| Ablation: sleep-only SSM | {sleep_s['f1']:.3f} | {sleep_s['auroc']:.3f} | {sleep_s['precision']:.3f} | {sleep_s['recall']:.3f} | {sleep_s['mean_detection_delay']} | {sleep_s['false_alert_rate']:.3f} |

Detection delays for proposed (per ramp crew {RAMP_CREW}): {prop['detection_delay_days']}

## Proposed model fidelity (vs simulated truth)

- Correlation(estimated fatigue, true fatigue): **{corr_all:.3f}**
- RMSE(fatigue): **{rmse_fat:.3f}**
- Calibrated fatigue alert threshold: {f_thresh:.3f}; P(elevated)>={ALERT_PROB_THRESH}

## Did the proposed method win?

- Beats baselines on F1: **{beats_f1}**
- Beats baselines on AUROC: **{beats_auc}**
- Detection delay competitive (not much worse than thresholds): **{delay_ok}**
- False alert rate (proposed): {prop['false_alert_rate']:.3f} vs thresholds {baseA['false_alert_rate']:.3f}, EWMA {baseB['false_alert_rate']:.3f}

## Ablation takeaway

Multimodal AUROC {prop['auroc']:.3f} vs HRV-only {hrv_s['auroc']:.3f} vs sleep-only {sleep_s['auroc']:.3f}.

## Limitations (state plainly)

1. **Synthetic data only** — dynamics, loadings, noise, and missingness are authored by us;
   real physiological coupling and operational constraints differ.
2. Approximate inference (Kalman smoother + hierarchical shrinkage / ridge), not full MCMC.
3. Ground-truth labels use the simulated latent fatigue threshold ({FATIGUE_GT_THRESH});
   real “elevated fatigue” has no such oracle.
4. Small N ({N_CREW} crew) and one random seed ({SEED}); results are illustrative, not a
   Monte Carlo uncertainty study.
5. Confound and ramp schedules are known by construction; operational deployment would
   face unlabeled change-points and distribution shift.

## Files

- Script: `simulate_and_evaluate.py`
- Metrics: `results/metrics.json`
- Figures: `figures/fig1_latent_vs_estimates.png`, `fig2_roc_comparison.png`,
  `fig3_delay_falsealarms.png`, `fig4_missingness_schematic.png`
"""
    summary_path = RESULTS / "summary.md"
    summary_path.write_text(summary_md)
    print(f"Wrote {summary_path}")

    print("\n--- Key metrics ---")
    for s in summaries:
        print(
            f"{s['method']:28s}  F1={s['f1']:.3f}  AUROC={s['auroc']:.3f}  "
            f"delay={s['mean_detection_delay']}  FAR={s['false_alert_rate']:.3f}"
        )
    print("Done.")
    return metrics


if __name__ == "__main__":
    main()
