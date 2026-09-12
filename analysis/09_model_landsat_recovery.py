#!/usr/bin/env python3
"""Audit functional form of event-level Landsat NBR recovery trajectories.

Inputs are already event-year aggregates. This script does not extract
Landsat, Sentinel-2, AlphaEarth, or site-level data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_EVENT_YEAR = ROOT / "outputs" / "reference_timeseries" / "disturbed_reference_event_year.csv"
PAIR_COMPLETENESS = ROOT / "outputs" / "reference_timeseries" / "reference_pair_completeness.csv"
OUT_DIR = ROOT / "outputs" / "landsat_recovery_model"
FIG_DIR = ROOT / "figures" / "landsat_recovery_model"
MPL_DIR = OUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer


FIXED_AGES = [0, 1, 5, 10, 15, 20]
RAW_FIXED_AGES = [0, 5, 10, 15, 20]
BOOTSTRAP_ITERATIONS = 600
RANDOM_SEED = 42
ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_year = pd.read_csv(INPUT_EVENT_YEAR)
    pairs = pd.read_csv(PAIR_COMPLETENESS)
    pairs = pairs[pairs["support_class"].eq("both_fire_and_harvest_supported")].copy()
    keep = set(pairs["fire_event_id"]).union(set(pairs["harvest_event_id"]))
    event_year = event_year[event_year["event_id"].isin(keep)].copy()
    event_year["is_harvest"] = event_year["disturbance_type"].eq("harvest_total").astype(int)
    event_year["calendar_year_centered"] = event_year["observation_year"] - event_year["observation_year"].mean()
    return event_year, pairs


def event_pre_baseline(event_year: pd.DataFrame) -> pd.DataFrame:
    pre = event_year[event_year["years_since_disturbance"].isin([-3, -2, -1])].copy()
    baseline = pre.groupby(["disturbance_type", "event_id", "disturbance_year"], as_index=False).agg(
        pre_nbr_gap=("nbr_gap_median", "median"),
        pre_ndvi_gap=("ndvi_gap_median", "median"),
        pre_year_count=("years_since_disturbance", "nunique"),
        pre_event_year_count=("observation_year", "count"),
        pre_supported_site_reference_pairs_median=("supported_site_reference_pairs", "median"),
    )
    baseline["abs_pre_nbr_gap"] = baseline["pre_nbr_gap"].abs()
    baseline["pre_gap_outlier_abs_gt_0_2"] = baseline["abs_pre_nbr_gap"] > 0.2
    return baseline


def build_longitudinal(event_year: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    data = event_year.merge(
        baseline[["event_id", "pre_nbr_gap", "pre_ndvi_gap", "pre_year_count", "pre_gap_outlier_abs_gt_0_2"]],
        on="event_id",
        how="inner",
    )
    data["nbr_gap_change"] = data["nbr_gap_median"] - data["pre_nbr_gap"]
    data["ndvi_gap_change"] = data["ndvi_gap_median"] - data["pre_ndvi_gap"]
    data["period"] = pd.cut(
        data["years_since_disturbance"],
        bins=[-999, -1, 0, 5, 10, 15, 20, 999],
        labels=["pre_disturbance", "year_0", "years_1_5", "years_6_10", "years_11_15", "years_16_20", "years_21plus"],
    )
    return data.sort_values(["disturbance_type", "event_id", "observation_year"])


def linear_design(df: pd.DataFrame, include_calendar_year: bool = False) -> tuple[np.ndarray, list[str]]:
    age = df["years_since_disturbance"].to_numpy(dtype=float)
    harvest = df["is_harvest"].to_numpy(dtype=float)
    cols = [np.ones(len(df)), age, harvest, age * harvest]
    names = ["intercept_fire", "years_since_disturbance", "harvest_main", "harvest_x_years_since"]
    if include_calendar_year:
        cols.append(df["calendar_year_centered"].to_numpy(dtype=float))
        names.append("calendar_year_centered")
    return np.column_stack(cols), names


def ols_fit(df: pd.DataFrame, response: str, include_calendar_year: bool = False) -> dict:
    work = df.dropna(subset=[response, "years_since_disturbance", "is_harvest"]).copy()
    x, names = linear_design(work, include_calendar_year)
    y = work[response].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    xtx_inv = np.linalg.pinv(x.T @ x)
    clusters = work["event_id"].astype(str).to_numpy()
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in np.unique(clusters):
        idx = clusters == cluster
        xg = x[idx]
        rg = resid[idx][:, None]
        meat += xg.T @ rg @ rg.T @ xg
    g = len(np.unique(clusters))
    n = len(work)
    k = x.shape[1]
    small = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    cov = small * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    t_stat = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stat), df=max(g - 1, 1))
    rss = float(np.sum(resid**2))
    aic = float(n * np.log(rss / n) + 2 * k) if rss > 0 and n > 0 else np.nan
    return {
        "data": work,
        "beta": beta,
        "cov": cov,
        "terms": names,
        "response": response,
        "include_calendar_year": include_calendar_year,
        "fitted": fitted,
        "resid": resid,
        "aic": aic,
        "coef": pd.DataFrame(
            {
                "term": names,
                "estimate": beta,
                "cluster_robust_se": se,
                "t": t_stat,
                "p": p_values,
                "n_event_years": n,
                "n_events": g,
            }
        ),
    }


def predict_linear(fit: dict, ages: list[int]) -> pd.DataFrame:
    rows = []
    for age in ages:
        for dtype, harvest in [("fire_total", 0), ("harvest_total", 1)]:
            cols = [1.0, float(age), float(harvest), float(age * harvest)]
            if fit["include_calendar_year"]:
                cols.append(0.0)
            x = np.array(cols)
            estimate = float(x @ fit["beta"])
            se = float(np.sqrt(np.clip(x @ fit["cov"] @ x.T, 0, np.inf)))
            rows.append(
                {
                    "model": fit["response"],
                    "disturbance_type": dtype,
                    "years_since_disturbance": age,
                    "estimate": estimate,
                    "cluster_robust_se": se,
                    "ci95_low": estimate - 1.96 * se,
                    "ci95_high": estimate + 1.96 * se,
                    "n_event_years": int(fit["coef"]["n_event_years"].iloc[0]),
                    "n_events": int(fit["coef"]["n_events"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def fit_spline_transformer(ages: np.ndarray) -> SplineTransformer:
    unique = np.unique(ages)
    n_knots = min(6, max(4, len(unique) // 4))
    transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False, extrapolation="constant")
    transformer.fit(ages.reshape(-1, 1))
    return transformer


def spline_matrix(df: pd.DataFrame, transformer: SplineTransformer) -> np.ndarray:
    age = df["years_since_disturbance"].to_numpy(dtype=float)
    harvest = df["is_harvest"].to_numpy(dtype=float)
    spline = transformer.transform(age.reshape(-1, 1))
    fire = spline * (1 - harvest[:, None])
    harvest_cols = spline * harvest[:, None]
    return np.column_stack([np.ones(len(df)), harvest, fire, harvest_cols])


def spline_matrix_for_grid(ages: np.ndarray, disturbance_type: str, transformer: SplineTransformer) -> np.ndarray:
    harvest = np.repeat(1.0 if disturbance_type == "harvest_total" else 0.0, len(ages))
    spline = transformer.transform(ages.reshape(-1, 1))
    return np.column_stack([np.ones(len(ages)), harvest, spline * (1 - harvest[:, None]), spline * harvest[:, None]])


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    if alpha == 0:
        alpha = 1e-10
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x, y)
    return model


def spline_edf(x: np.ndarray, alpha: float) -> float:
    penalty = np.eye(x.shape[1]) * max(alpha, 1e-10)
    penalty[0, 0] = 0
    hat = x @ np.linalg.pinv(x.T @ x + penalty) @ x.T
    return float(np.trace(hat))


def grouped_cv_compare(df: pd.DataFrame, response: str) -> tuple[pd.DataFrame, float]:
    work = df.dropna(subset=[response]).copy()
    groups = work["event_id"].astype(str).to_numpy()
    splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=splits)
    rows = []
    linear_pred = np.full(len(work), np.nan)
    for train, test in gkf.split(work, groups=groups):
        fit = ols_fit(work.iloc[train], response)
        xt, _ = linear_design(work.iloc[test])
        linear_pred[test] = xt @ fit["beta"]
    y = work[response].to_numpy(dtype=float)
    rows.append(
        {
            "response": response,
            "model": "linear",
            "alpha": 0.0,
            "cv_rmse": float(mean_squared_error(y, linear_pred) ** 0.5),
            "cv_mae": float(mean_absolute_error(y, linear_pred)),
            "n_event_years": int(len(work)),
            "n_events": int(len(np.unique(groups))),
            "effective_degrees_of_freedom": 4.0,
            "selection_note": "simple comparator",
        }
    )
    best_alpha = ALPHAS[0]
    best_rmse = np.inf
    for alpha in ALPHAS:
        pred = np.full(len(work), np.nan)
        for train, test in gkf.split(work, groups=groups):
            transformer = fit_spline_transformer(work.iloc[train]["years_since_disturbance"].to_numpy(dtype=float))
            x_train = spline_matrix(work.iloc[train], transformer)
            x_test = spline_matrix(work.iloc[test], transformer)
            model = ridge_fit(x_train, work.iloc[train][response].to_numpy(dtype=float), alpha)
            pred[test] = model.predict(x_test)
        rmse = float(mean_squared_error(y, pred) ** 0.5)
        rows.append(
            {
                "response": response,
                "model": "nonlinear_spline",
                "alpha": alpha,
                "cv_rmse": rmse,
                "cv_mae": float(mean_absolute_error(y, pred)),
                "n_event_years": int(len(work)),
                "n_events": int(len(np.unique(groups))),
                "effective_degrees_of_freedom": np.nan,
                "selection_note": "candidate smoothing penalty",
            }
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return pd.DataFrame(rows), best_alpha


def fit_spline_model(df: pd.DataFrame, response: str, alpha: float) -> dict:
    work = df.dropna(subset=[response]).copy()
    transformer = fit_spline_transformer(work["years_since_disturbance"].to_numpy(dtype=float))
    x = spline_matrix(work, transformer)
    y = work[response].to_numpy(dtype=float)
    model = ridge_fit(x, y, alpha)
    fitted = model.predict(x)
    resid = y - fitted
    rss = float(np.sum(resid**2))
    edf = spline_edf(x, alpha)
    aic = float(len(work) * np.log(rss / len(work)) + 2 * edf) if rss > 0 else np.nan
    return {"data": work, "transformer": transformer, "model": model, "response": response, "alpha": alpha, "fitted": fitted, "resid": resid, "edf": edf, "aic": aic}


def bootstrap_spline_predictions(fit: dict, ages: list[int] | np.ndarray, iterations: int = BOOTSTRAP_ITERATIONS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    work = fit["data"].copy()
    ages = np.array(ages, dtype=float)
    rows = []
    point_rows = []
    for dtype in ["fire_total", "harvest_total"]:
        xg = spline_matrix_for_grid(ages, dtype, fit["transformer"])
        pred = fit["model"].predict(xg)
        for age, estimate in zip(ages, pred):
            point_rows.append({"disturbance_type": dtype, "years_since_disturbance": int(age), "estimate": float(estimate)})
    point = pd.DataFrame(point_rows)
    boot = []
    events = work["event_id"].drop_duplicates().to_numpy()
    for _ in range(iterations):
        sampled_events = rng.choice(events, size=len(events), replace=True)
        parts = []
        for i, event_id in enumerate(sampled_events):
            part = work[work["event_id"].eq(event_id)].copy()
            part["event_id"] = f"{event_id}__boot{i}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            model = fit_spline_model(sample, fit["response"], fit["alpha"])
        except Exception:
            continue
        pred_rows = []
        for dtype in ["fire_total", "harvest_total"]:
            xg = spline_matrix_for_grid(ages, dtype, model["transformer"])
            pred = model["model"].predict(xg)
            for age, value in zip(ages, pred):
                pred_rows.append((dtype, int(age), float(value)))
        boot.append(pred_rows)
    boot_df = pd.DataFrame([{"disturbance_type": d, "years_since_disturbance": a, "value": v} for sample in boot for d, a, v in sample])
    for _, row in point.iterrows():
        vals = boot_df[(boot_df["disturbance_type"].eq(row["disturbance_type"])) & (boot_df["years_since_disturbance"].eq(row["years_since_disturbance"]))]["value"]
        rows.append(
            {
                "model": fit["response"],
                "disturbance_type": row["disturbance_type"],
                "years_since_disturbance": int(row["years_since_disturbance"]),
                "estimate": row["estimate"],
                "ci95_low": float(vals.quantile(0.025)) if len(vals) else np.nan,
                "ci95_high": float(vals.quantile(0.975)) if len(vals) else np.nan,
                "bootstrap_iterations": int(len(vals)),
                "alpha": fit["alpha"],
                "effective_degrees_of_freedom": fit["edf"],
            }
        )
    return pd.DataFrame(rows)


def nonlinear_difference_curve(pred: pd.DataFrame) -> pd.DataFrame:
    fire = pred[pred["disturbance_type"].eq("fire_total")].set_index("years_since_disturbance")
    harvest = pred[pred["disturbance_type"].eq("harvest_total")].set_index("years_since_disturbance")
    rows = []
    # Conservative CI from marginal bands. Bootstrap paired bands are not retained.
    for age in sorted(set(fire.index).intersection(harvest.index)):
        f = fire.loc[age]
        h = harvest.loc[age]
        diff = f["estimate"] - h["estimate"]
        low = f["ci95_low"] - h["ci95_high"]
        high = f["ci95_high"] - h["ci95_low"]
        rows.append(
            {
                "years_since_disturbance": int(age),
                "fire_minus_harvest": float(diff),
                "ci95_low_conservative": float(low),
                "ci95_high_conservative": float(high),
                "ci_overlaps_zero": bool(low <= 0 <= high),
            }
        )
    return pd.DataFrame(rows)


def fixed_age_contrasts(pred: pd.DataFrame) -> pd.DataFrame:
    diff = nonlinear_difference_curve(pred)
    diff = diff[diff["years_since_disturbance"].isin(FIXED_AGES)].copy()
    fire = pred[pred["disturbance_type"].eq("fire_total")][["years_since_disturbance", "estimate"]].rename(columns={"estimate": "fire_estimate"})
    harvest = pred[pred["disturbance_type"].eq("harvest_total")][["years_since_disturbance", "estimate"]].rename(columns={"estimate": "harvest_estimate"})
    out = diff.merge(fire, on="years_since_disturbance").merge(harvest, on="years_since_disturbance")
    return out[["years_since_disturbance", "fire_estimate", "harvest_estimate", "fire_minus_harvest", "ci95_low_conservative", "ci95_high_conservative", "ci_overlaps_zero"]]


def interval_changes(pred: pd.DataFrame, response_label: str) -> pd.DataFrame:
    rows = []
    for dtype, group in pred.groupby("disturbance_type"):
        lookup = group.set_index("years_since_disturbance")
        for start, end in [(0, 5), (5, 10), (10, 15), (15, 20)]:
            if start in lookup.index and end in lookup.index:
                rows.append(
                    {
                        "response": response_label,
                        "disturbance_type": dtype,
                        "interval": f"{start}->{end}",
                        "start_age": start,
                        "end_age": end,
                        "change": float(lookup.loc[end, "estimate"] - lookup.loc[start, "estimate"]),
                    }
                )
    return pd.DataFrame(rows)


def descriptive_by_age(data: pd.DataFrame) -> pd.DataFrame:
    post = data[data["years_since_disturbance"].ge(0)].copy()
    return post.groupby(["disturbance_type", "years_since_disturbance"], as_index=False).agg(
        event_years=("event_id", "count"),
        events=("event_id", "nunique"),
        median_nbr_gap=("nbr_gap_median", "median"),
        mean_nbr_gap=("nbr_gap_median", "mean"),
        sd_nbr_gap=("nbr_gap_median", "std"),
        median_nbr_gap_change=("nbr_gap_change", "median"),
        mean_nbr_gap_change=("nbr_gap_change", "mean"),
        sd_nbr_gap_change=("nbr_gap_change", "std"),
        median_dist_nbr=("dist_nbr_median", "median"),
        median_ref_nbr=("ref_nbr_median", "median"),
        median_supported_site_reference_pairs=("supported_site_reference_pairs", "median"),
    )


def paired_recovery_differences(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    keep = pairs[["pair_id", "fire_event_id", "harvest_event_id"]].copy()
    fire = data[data["disturbance_type"].eq("fire_total")].copy().rename(columns={"event_id": "fire_event_id"})
    harvest = data[data["disturbance_type"].eq("harvest_total")].copy().rename(columns={"event_id": "harvest_event_id"})
    fire = keep.merge(fire, on="fire_event_id", how="inner")
    merged = fire.merge(harvest, on=["harvest_event_id", "observation_year", "years_since_disturbance"], suffixes=("_fire", "_harvest"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "pair_id": row["pair_id"],
                "fire_event_id": row["fire_event_id"],
                "harvest_event_id": row["harvest_event_id"],
                "observation_year": int(row["observation_year"]),
                "years_since_disturbance": int(row["years_since_disturbance"]),
                "fire_nbr_gap_change": row["nbr_gap_change_fire"],
                "harvest_nbr_gap_change": row["nbr_gap_change_harvest"],
                "fire_minus_harvest_nbr_gap_change": row["nbr_gap_change_fire"] - row["nbr_gap_change_harvest"],
                "fire_nbr_gap": row["nbr_gap_median_fire"],
                "harvest_nbr_gap": row["nbr_gap_median_harvest"],
                "fire_minus_harvest_nbr_gap": row["nbr_gap_median_fire"] - row["nbr_gap_median_harvest"],
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_id", "observation_year"])


def paired_spline_predictions(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = paired[paired["years_since_disturbance"].ge(0)].dropna(subset=["fire_minus_harvest_nbr_gap_change"]).copy()
    if post.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = post.rename(columns={"pair_id": "event_id"}).copy()
    work["disturbance_type"] = "paired_difference"
    work["is_harvest"] = 0
    cv_rows = []
    groups = work["event_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    best_alpha, best_rmse = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        pred = np.full(len(work), np.nan)
        for train, test in gkf.split(work, groups=groups):
            transformer = fit_spline_transformer(work.iloc[train]["years_since_disturbance"].to_numpy(dtype=float))
            x_train = transformer.transform(work.iloc[train]["years_since_disturbance"].to_numpy(dtype=float).reshape(-1, 1))
            x_train = np.column_stack([np.ones(len(x_train)), x_train])
            x_test = transformer.transform(work.iloc[test]["years_since_disturbance"].to_numpy(dtype=float).reshape(-1, 1))
            x_test = np.column_stack([np.ones(len(x_test)), x_test])
            model = ridge_fit(x_train, work.iloc[train]["fire_minus_harvest_nbr_gap_change"].to_numpy(dtype=float), alpha)
            pred[test] = model.predict(x_test)
        rmse = float(mean_squared_error(work["fire_minus_harvest_nbr_gap_change"], pred) ** 0.5)
        cv_rows.append({"model": "paired_difference_spline", "alpha": alpha, "cv_rmse": rmse, "cv_mae": float(mean_absolute_error(work["fire_minus_harvest_nbr_gap_change"], pred)), "n_pair_years": len(work), "n_pairs": len(np.unique(groups))})
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha
    transformer = fit_spline_transformer(work["years_since_disturbance"].to_numpy(dtype=float))
    x = transformer.transform(work["years_since_disturbance"].to_numpy(dtype=float).reshape(-1, 1))
    x = np.column_stack([np.ones(len(x)), x])
    model = ridge_fit(x, work["fire_minus_harvest_nbr_gap_change"].to_numpy(dtype=float), best_alpha)
    rows = []
    for age in FIXED_AGES:
        xp = transformer.transform(np.array([[age]], dtype=float))
        estimate = float(model.predict(np.column_stack([np.ones(len(xp)), xp]))[0])
        rows.append({"years_since_disturbance": age, "fire_minus_harvest_nbr_gap_change": estimate, "alpha": best_alpha, "n_pair_years": len(work), "n_pairs": len(np.unique(groups))})
    return pd.DataFrame(rows), pd.DataFrame(cv_rows)


def paired_support_by_age(paired: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    ages = sorted(data.loc[data["years_since_disturbance"].ge(0), "years_since_disturbance"].unique())
    rows = []
    for age in ages:
        f = data[(data["disturbance_type"].eq("fire_total")) & (data["years_since_disturbance"].eq(age))]
        h = data[(data["disturbance_type"].eq("harvest_total")) & (data["years_since_disturbance"].eq(age))]
        p = paired[(paired["years_since_disturbance"].eq(age)) & paired["fire_minus_harvest_nbr_gap_change"].notna()]
        rows.append(
            {
                "years_since_disturbance": int(age),
                "n_fire_events": int(f["event_id"].nunique()),
                "n_harvest_events": int(h["event_id"].nunique()),
                "n_complete_fire_harvest_pairs": int(p["pair_id"].nunique()),
                "n_fire_event_years": int(len(f)),
                "n_harvest_event_years": int(len(h)),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_shape(data: pd.DataFrame, response: str) -> pd.DataFrame:
    rows = []
    specs = [
        ("all_observations", data[data["years_since_disturbance"].ge(0)]),
        ("exclude_calendar_year_2022", data[data["years_since_disturbance"].ge(0) & data["observation_year"].ne(2022)]),
        ("exclude_abs_pre_nbr_gap_gt_0_2", data[data["years_since_disturbance"].ge(0) & ~data["pre_gap_outlier_abs_gt_0_2"]]),
    ]
    for name, df in specs:
        comp, alpha = grouped_cv_compare(df, response)
        comp["specification"] = name
        comp["selected_alpha"] = alpha
        rows.append(comp)
    return pd.concat(rows, ignore_index=True)


def model_diagnostics(fit: dict, grid_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = fit["data"].copy()
    work["fitted"] = fit["fitted"]
    work["residual"] = fit["resid"]
    rows = [
        {"metric": "residual_mean", "value": float(work["residual"].mean())},
        {"metric": "residual_sd", "value": float(work["residual"].std())},
        {"metric": "residual_age_correlation", "value": float(work[["residual", "years_since_disturbance"]].corr().iloc[0, 1])},
        {"metric": "abs_residual_p95", "value": float(work["residual"].abs().quantile(0.95))},
        {"metric": "effective_degrees_of_freedom", "value": float(fit["edf"])},
        {"metric": "aic_approx_gaussian", "value": float(fit["aic"])},
    ]
    full = grid_pred.set_index(["disturbance_type", "years_since_disturbance"])["estimate"]
    influence = []
    for event_id in work["event_id"].drop_duplicates():
        reduced = work[~work["event_id"].eq(event_id)].copy()
        if reduced["event_id"].nunique() < 10:
            continue
        try:
            refit = fit_spline_model(reduced, fit["response"], fit["alpha"])
            pred = bootstrap_spline_predictions(refit, FIXED_AGES, iterations=0)
        except Exception:
            continue
        reduced_pred = pred.set_index(["disturbance_type", "years_since_disturbance"])["estimate"]
        common = full.index.intersection(reduced_pred.index)
        max_delta = float((reduced_pred.loc[common] - full.loc[common]).abs().max())
        influence.append({"event_id": event_id, "disturbance_type": work.loc[work["event_id"].eq(event_id), "disturbance_type"].iloc[0], "max_abs_prediction_change_fixed_ages": max_delta})
    influence_df = pd.DataFrame(influence)
    if influence_df.empty:
        influence_df = pd.DataFrame(
            columns=["event_id", "disturbance_type", "max_abs_prediction_change_fixed_ages"]
        )
    else:
        influence_df = influence_df.sort_values("max_abs_prediction_change_fixed_ages", ascending=False)
    return pd.DataFrame(rows), influence_df


def write_figures(
    data: pd.DataFrame,
    linear_pred: pd.DataFrame,
    nonlinear_pred: pd.DataFrame,
    diff_curve: pd.DataFrame,
    raw_pred: pd.DataFrame,
    paired: pd.DataFrame,
    paired_pred: pd.DataFrame,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_fit: dict,
) -> None:
    colors = {"fire_total": "#c95f3f", "harvest_total": "#2f7f73"}
    desc = descriptive_by_age(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for dtype, group in desc.groupby("disturbance_type"):
        ax.scatter(group["years_since_disturbance"], group["median_nbr_gap_change"], s=24, color=colors[dtype], alpha=0.55)
    for dtype, group in nonlinear_pred.groupby("disturbance_type"):
        ax.plot(group["years_since_disturbance"], group["estimate"], color=colors[dtype], label=dtype)
        ax.fill_between(group["years_since_disturbance"], group["ci95_low"], group["ci95_high"], color=colors[dtype], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("NBR gap change")
    ax.set_title("Nonlinear reference-relative NBR recovery")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nonlinear_fire_harvest_recovery.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for dtype, group in nonlinear_pred.groupby("disturbance_type"):
        ax.plot(group["years_since_disturbance"], group["estimate"], color=colors[dtype], label=f"{dtype} nonlinear")
    for dtype, group in linear_pred.groupby("disturbance_type"):
        ax.plot(group["years_since_disturbance"], group["estimate"], color=colors[dtype], linestyle="--", alpha=0.75, label=f"{dtype} linear")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("NBR gap change")
    ax.set_title("Linear vs nonlinear functional form")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "linear_vs_nonlinear_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(diff_curve["years_since_disturbance"], diff_curve["fire_minus_harvest"], color="#143d36")
    ax.fill_between(diff_curve["years_since_disturbance"], diff_curve["ci95_low_conservative"], diff_curve["ci95_high_conservative"], color="#143d36", alpha=0.16)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("fire - harvest NBR gap change")
    ax.set_title("Nonlinear fire-minus-harvest difference")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nonlinear_fire_minus_harvest.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    raw_desc = data[data["years_since_disturbance"].ge(0)].groupby(["disturbance_type", "years_since_disturbance"], as_index=False).agg(median_nbr_gap=("nbr_gap_median", "median"))
    for dtype, group in raw_desc.groupby("disturbance_type"):
        ax.scatter(group["years_since_disturbance"], group["median_nbr_gap"], s=22, color=colors[dtype], alpha=0.5)
    for dtype, group in raw_pred.groupby("disturbance_type"):
        ax.plot(group["years_since_disturbance"], group["estimate"], color=colors[dtype], label=dtype)
        ax.fill_between(group["years_since_disturbance"], group["ci95_low"], group["ci95_high"], color=colors[dtype], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("raw NBR gap")
    ax.set_title("Raw NBR-gap convergence toward reference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_nbr_gap_convergence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if not paired.empty:
        pair_desc = paired[paired["years_since_disturbance"].ge(0)].groupby("years_since_disturbance", as_index=False).agg(median_diff=("fire_minus_harvest_nbr_gap_change", "median"))
        ax.scatter(pair_desc["years_since_disturbance"], pair_desc["median_diff"], color="#526760", alpha=0.65)
    if not paired_pred.empty:
        ax.plot(paired_pred["years_since_disturbance"], paired_pred["fire_minus_harvest_nbr_gap_change"], color="#143d36")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("paired fire - harvest NBR gap change")
    ax.set_title("Paired-difference nonlinear trajectory")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paired_difference_nonlinear.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(support["years_since_disturbance"], support["n_fire_events"], marker="o", color=colors["fire_total"], label="fire events")
    ax.plot(support["years_since_disturbance"], support["n_harvest_events"], marker="o", color=colors["harvest_total"], label="harvest events")
    ax.plot(support["years_since_disturbance"], support["n_complete_fire_harvest_pairs"], marker="o", color="#202020", label="complete pairs")
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("event/pair support")
    ax.set_title("Event and exact-pair support by age")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_support_by_age.png", dpi=220)
    plt.close(fig)

    work = selected_fit["data"].copy()
    work["fitted"] = selected_fit["fitted"]
    work["residual"] = selected_fit["resid"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(work["fitted"], work["residual"], s=10, alpha=0.35)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("fitted")
    axes[0].set_ylabel("residual")
    axes[0].set_title("Residuals vs fitted")
    axes[1].scatter(work["years_since_disturbance"], work["residual"], s=10, alpha=0.35)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("age")
    axes[1].set_title("Residuals vs age")
    axes[2].hist(work["residual"], bins=28, color="#526760", edgecolor="white")
    axes[2].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_diagnostics.png", dpi=220)
    plt.close(fig)


def write_readme(summary: dict) -> None:
    text = f"""# Landsat recovery model

This phase models event-level Landsat NBR recovery using
`outputs/reference_timeseries/disturbed_reference_event_year.csv`.

The primary response is:

`nbr_gap_change = (NBR_reference - NBR_disturbed) - pre_nbr_gap`

where `pre_nbr_gap` is the event-level median NBR gap from available years -3,
-2, and -1. The final primary model is a spline-based nonlinear trajectory
model with group/event cross-validation and event-bootstrap prediction
intervals. The linear model is retained as a simple comparator.

No new Landsat, Sentinel-2, or AlphaEarth data were extracted in this phase.

```json
{json.dumps(summary, indent=2)}
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    event_year, pairs = load_inputs()
    baseline = event_pre_baseline(event_year)
    data = build_longitudinal(event_year, baseline)
    post = data[data["years_since_disturbance"].ge(0)].copy()
    max_supported_age = int(post["years_since_disturbance"].max())
    age_grid = np.arange(0, max_supported_age + 1)

    linear_fit = ols_fit(post, "nbr_gap_change")
    raw_linear_fit = ols_fit(post, "nbr_gap_median")
    linear_pred_grid = predict_linear(linear_fit, age_grid.tolist())
    linear_fixed_pred = predict_linear(linear_fit, FIXED_AGES)
    raw_linear_pred = predict_linear(raw_linear_fit, RAW_FIXED_AGES)

    form_primary, best_alpha = grouped_cv_compare(post, "nbr_gap_change")
    form_raw, raw_best_alpha = grouped_cv_compare(post, "nbr_gap_median")
    nonlinear_fit = fit_spline_model(post, "nbr_gap_change", best_alpha)
    raw_nonlinear_fit = fit_spline_model(post, "nbr_gap_median", raw_best_alpha)
    nonlinear_pred_grid = bootstrap_spline_predictions(nonlinear_fit, age_grid)
    nonlinear_fixed = nonlinear_pred_grid[nonlinear_pred_grid["years_since_disturbance"].isin(FIXED_AGES)].copy()
    raw_pred = bootstrap_spline_predictions(raw_nonlinear_fit, RAW_FIXED_AGES)
    diff_curve = nonlinear_difference_curve(nonlinear_pred_grid)
    fixed_contrasts = fixed_age_contrasts(nonlinear_fixed)
    intervals = interval_changes(nonlinear_fixed, "nbr_gap_change")

    paired = paired_recovery_differences(data, pairs)
    paired_pred, paired_cv = paired_spline_predictions(paired)
    pair_support = paired_support_by_age(paired, data)
    sensitivity = sensitivity_shape(data, "nbr_gap_change")
    support = pair_support.copy()
    diagnostics, influence = model_diagnostics(nonlinear_fit, nonlinear_fixed)

    comparison = pd.concat([form_primary, form_raw], ignore_index=True)
    comparison["aic_approx_gaussian"] = np.nan
    comparison.loc[comparison["response"].eq("nbr_gap_change") & comparison["model"].eq("linear"), "aic_approx_gaussian"] = linear_fit["aic"]
    comparison.loc[comparison["response"].eq("nbr_gap_change") & comparison["model"].eq("nonlinear_spline") & comparison["alpha"].eq(best_alpha), "aic_approx_gaussian"] = nonlinear_fit["aic"]
    comparison.loc[comparison["response"].eq("nbr_gap_median") & comparison["model"].eq("linear"), "aic_approx_gaussian"] = raw_linear_fit["aic"]
    comparison.loc[comparison["response"].eq("nbr_gap_median") & comparison["model"].eq("nonlinear_spline") & comparison["alpha"].eq(raw_best_alpha), "aic_approx_gaussian"] = raw_nonlinear_fit["aic"]
    comparison.loc[comparison["response"].eq("nbr_gap_change") & comparison["model"].eq("nonlinear_spline") & comparison["alpha"].eq(best_alpha), "effective_degrees_of_freedom"] = nonlinear_fit["edf"]
    comparison.loc[comparison["response"].eq("nbr_gap_median") & comparison["model"].eq("nonlinear_spline") & comparison["alpha"].eq(raw_best_alpha), "effective_degrees_of_freedom"] = raw_nonlinear_fit["edf"]

    baseline.to_csv(OUT_DIR / "event_pre_baseline.csv", index=False, float_format="%.6f")
    data.to_csv(OUT_DIR / "event_recovery_longitudinal.csv", index=False, float_format="%.6f")
    descriptive_by_age(data).to_csv(OUT_DIR / "recovery_descriptive_by_age.csv", index=False, float_format="%.6f")
    linear_fixed_pred.to_csv(OUT_DIR / "primary_model_predictions.csv", index=False, float_format="%.6f")
    fixed_contrasts.to_csv(OUT_DIR / "fixed_age_contrasts.csv", index=False, float_format="%.6f")
    paired.to_csv(OUT_DIR / "paired_recovery_differences.csv", index=False, float_format="%.6f")
    paired_pred.to_csv(OUT_DIR / "paired_model_predictions.csv", index=False, float_format="%.6f")
    sensitivity[sensitivity["specification"].isin(["all_observations", "exclude_calendar_year_2022"])].to_csv(OUT_DIR / "calendar_year_sensitivity.csv", index=False, float_format="%.6f")
    sensitivity[sensitivity["specification"].isin(["all_observations", "exclude_abs_pre_nbr_gap_gt_0_2"])].to_csv(OUT_DIR / "pre_gap_outlier_sensitivity.csv", index=False, float_format="%.6f")
    support.to_csv(OUT_DIR / "event_support_by_age.csv", index=False, float_format="%.6f")
    raw_linear_pred.to_csv(OUT_DIR / "raw_nbr_gap_model_predictions.csv", index=False, float_format="%.6f")
    linear_fit["coef"].to_csv(OUT_DIR / "primary_model_coefficients.csv", index=False, float_format="%.6f")
    raw_linear_fit["coef"].to_csv(OUT_DIR / "raw_nbr_gap_model_coefficients.csv", index=False, float_format="%.6f")

    comparison.to_csv(OUT_DIR / "functional_form_comparison.csv", index=False, float_format="%.6f")
    nonlinear_pred_grid.to_csv(OUT_DIR / "nonlinear_model_predictions.csv", index=False, float_format="%.6f")
    fixed_contrasts.to_csv(OUT_DIR / "nonlinear_fixed_age_contrasts.csv", index=False, float_format="%.6f")
    diff_curve.to_csv(OUT_DIR / "nonlinear_fire_harvest_difference.csv", index=False, float_format="%.6f")
    raw_pred.to_csv(OUT_DIR / "raw_nbr_gap_predictions.csv", index=False, float_format="%.6f")
    pair_support.to_csv(OUT_DIR / "paired_support_by_age.csv", index=False, float_format="%.6f")
    sensitivity.to_csv(OUT_DIR / "nonlinear_sensitivity.csv", index=False, float_format="%.6f")
    diagnostics.to_csv(OUT_DIR / "model_diagnostics_summary.csv", index=False, float_format="%.6f")
    influence.to_csv(OUT_DIR / "event_influence_diagnostics.csv", index=False, float_format="%.6f")
    intervals.to_csv(OUT_DIR / "nonlinear_interval_changes.csv", index=False, float_format="%.6f")
    paired_cv.to_csv(OUT_DIR / "paired_functional_form_comparison.csv", index=False, float_format="%.6f")

    first_overlap = diff_curve.loc[diff_curve["ci_overlaps_zero"], "years_since_disturbance"]
    first_overlap_age = int(first_overlap.min()) if len(first_overlap) else None
    older_obs = post[post["years_since_disturbance"].ge(18)]
    negative_late_support = older_obs.groupby("disturbance_type")["nbr_gap_change"].agg(
        event_years="count",
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        proportion_negative=lambda s: float((s < 0).mean()),
    ).reset_index()
    negative_late_support.to_csv(OUT_DIR / "negative_late_age_assessment.csv", index=False, float_format="%.6f")

    best_linear = comparison[(comparison["response"].eq("nbr_gap_change")) & (comparison["model"].eq("linear"))].iloc[0]
    best_spline = comparison[(comparison["response"].eq("nbr_gap_change")) & (comparison["model"].eq("nonlinear_spline")) & (comparison["alpha"].eq(best_alpha))].iloc[0]
    selected = "nonlinear_spline" if best_spline["cv_rmse"] <= best_linear["cv_rmse"] * 1.01 else "linear"
    summary = {
        "input": str(INPUT_EVENT_YEAR.relative_to(ROOT)),
        "complete_exact_year_fire_harvest_pairs": int(len(pairs)),
        "events_with_pre_baseline": baseline.groupby("disturbance_type")["event_id"].nunique().to_dict(),
        "post_model_events": post.groupby("disturbance_type")["event_id"].nunique().to_dict(),
        "post_model_event_years": int(len(post)),
        "functional_form_selected_primary": selected,
        "linear_cv_rmse": float(best_linear["cv_rmse"]),
        "nonlinear_cv_rmse": float(best_spline["cv_rmse"]),
        "nonlinear_selected_alpha": float(best_alpha),
        "nonlinear_effective_degrees_of_freedom": float(nonlinear_fit["edf"]),
        "fire_harvest_ci_first_overlaps_zero_age": first_overlap_age,
        "nonlinear_fixed_age_fire_minus_harvest": fixed_contrasts.to_dict(orient="records"),
        "raw_nbr_gap_fixed_age_predictions": raw_pred.to_dict(orient="records"),
        "paired_model_pairs": int(paired[paired["years_since_disturbance"].ge(0)]["pair_id"].nunique()),
        "paired_support_fixed_ages": pair_support[pair_support["years_since_disturbance"].isin(FIXED_AGES)].to_dict(orient="records"),
        "negative_late_age_observation_support": negative_late_support.to_dict(orient="records"),
        "method_note": "Spline trajectories use event-level rows, event-grouped cross-validation, and event-bootstrap prediction intervals; linear model remains the simple comparator.",
    }
    (OUT_DIR / "landsat_recovery_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    write_figures(data, linear_pred_grid, nonlinear_pred_grid, diff_curve, raw_pred, paired, paired_pred, pair_support, diagnostics, nonlinear_fit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
