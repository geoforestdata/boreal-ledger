#!/usr/bin/env python3
"""Phase 11: model multivariate recovery trajectories from Phase 10 outputs.

This phase uses event-year tables only. It does not extract imagery, rematch
events, or treat site-years as independent disturbance replicates.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / "outputs" / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

IN_DIR = ROOT / "outputs" / "multivariate_recovery"
LANDSAT_DIR = ROOT / "outputs" / "landsat_recovery_model"
OUT_DIR = ROOT / "outputs" / "multivariate_recovery_model"
FIG_DIR = ROOT / "figures" / "multivariate_recovery_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIXED_AGES = [0, 1, 5, 10, 15, 20]
PRIMARY_AGES = [0, 5, 10, 15]
INTERVALS = [(0, 5), (5, 10), (10, 15), (15, 20)]
N_BOOT = 300
RNG = np.random.default_rng(20260911)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    multi = pd.read_csv(IN_DIR / "multivariate_event_year.csv")
    support = pd.read_csv(IN_DIR / "temporal_support_by_age.csv")
    multi["product_pair_available"] = multi["both_products_available"].astype(bool)
    landsat = pd.read_csv(LANDSAT_DIR / "event_recovery_longitudinal.csv")
    return multi, support


def attach_event_pairs_for_paired_analysis(df: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(ROOT / "outputs" / "reference_timeseries" / "reference_pair_completeness.csv")
    fire = df[df["disturbance_type"] == "fire_total"].merge(
        pairs[["pair_id", "fire_event_id"]],
        left_on="event_id",
        right_on="fire_event_id",
        how="inner",
    ).drop(columns=["fire_event_id"])
    harvest = df[df["disturbance_type"] == "harvest_total"].merge(
        pairs[["pair_id", "harvest_event_id"]],
        left_on="event_id",
        right_on="harvest_event_id",
        how="inner",
    ).drop(columns=["harvest_event_id"])
    out = pd.concat([fire, harvest], ignore_index=True).rename(columns={"pair_id": "event_pair_id"})
    return out


def prep_product(multi: pd.DataFrame, product: str, metric: str, min_s2_sites: int = 1) -> pd.DataFrame:
    cols = ["disturbance_type", "event_id", "disturbance_year", "observation_year", "years_since_disturbance", metric]
    if product == "sentinel2":
        cols.append("n_valid_sites_s2")
        df = multi[cols].rename(columns={"n_valid_sites_s2": "n_valid_sites", metric: "distance"}).copy()
        df = df[df["n_valid_sites"].fillna(0) >= min_s2_sites]
    else:
        cols.append("n_valid_sites_ae")
        df = multi[cols].rename(columns={"n_valid_sites_ae": "n_valid_sites", metric: "distance"}).copy()
        df = df[df["n_valid_sites"].fillna(0) >= 1]
    df = df[df["distance"].notna()].copy()
    df["product"] = product
    return df


def spline_features(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    age_cap = np.minimum(age, 15)
    tail = np.maximum(age - 15, 0)
    spline = SplineTransformer(
        degree=3,
        knots=np.array([[0.0], [5.0], [10.0], [15.0]]),
        include_bias=False,
        extrapolation="constant",
    )
    x = spline.fit_transform(age_cap)
    return np.column_stack([x, tail])


def design_flexible(df: pd.DataFrame) -> np.ndarray:
    x = spline_features(df["years_since_disturbance"].to_numpy())
    is_h = (df["disturbance_type"] == "harvest_total").astype(float).to_numpy().reshape(-1, 1)
    return np.column_stack([np.ones(len(df)), is_h, x, x * is_h])


def design_linear(df: pd.DataFrame) -> np.ndarray:
    age = df["years_since_disturbance"].to_numpy(dtype=float)
    is_h = (df["disturbance_type"] == "harvest_total").astype(float).to_numpy()
    return np.column_stack([np.ones(len(df)), age, is_h, age * is_h])


def pred_frame(ages: list[int] | np.ndarray) -> pd.DataFrame:
    rows = []
    for d in ["fire_total", "harvest_total"]:
        for a in ages:
            rows.append({"disturbance_type": d, "years_since_disturbance": float(a)})
    return pd.DataFrame(rows)


def fit_predict(df: pd.DataFrame, metric_name: str, product: str, ages: list[int] = FIXED_AGES) -> tuple[pd.DataFrame, pd.DataFrame, Ridge]:
    model_df = df[(df["years_since_disturbance"] >= 0) & (df["years_since_disturbance"] <= 20)].copy()
    pred = pred_frame(ages)
    x = design_flexible(model_df)
    y = model_df["distance"].to_numpy()
    model = Ridge(alpha=1e-6, fit_intercept=False).fit(x, y)
    pred_vals = model.predict(design_flexible(pred))

    event_ids = model_df["event_id"].unique()
    boot = []
    for _ in range(N_BOOT):
        sampled = RNG.choice(event_ids, size=len(event_ids), replace=True)
        bdf = pd.concat([model_df[model_df["event_id"] == e] for e in sampled], ignore_index=True)
        if bdf["disturbance_type"].nunique() < 2:
            continue
        try:
            bm = Ridge(alpha=1e-6, fit_intercept=False).fit(design_flexible(bdf), bdf["distance"].to_numpy())
            boot.append(bm.predict(design_flexible(pred)))
        except Exception:
            continue
    boot = np.asarray(boot)
    out = pred.copy()
    out.insert(0, "product", product)
    out.insert(1, "metric", metric_name)
    out["estimate"] = pred_vals
    out["ci95_low"] = np.nanpercentile(boot, 2.5, axis=0)
    out["ci95_high"] = np.nanpercentile(boot, 97.5, axis=0)
    support = model_df.groupby(["disturbance_type", "years_since_disturbance"]).agg(
        n_event_years=("event_id", "size"), n_events=("event_id", "nunique")
    ).reset_index()
    out = out.merge(support, on=["disturbance_type", "years_since_disturbance"], how="left")

    changes = []
    available_ages = set(out["years_since_disturbance"].astype(float))
    for d in ["fire_total", "harvest_total"]:
        for a0, a1 in INTERVALS:
            if float(a0) not in available_ages or float(a1) not in available_ages:
                continue
            p0 = out[(out.disturbance_type == d) & (out.years_since_disturbance == a0)]["estimate"].iloc[0]
            p1 = out[(out.disturbance_type == d) & (out.years_since_disturbance == a1)]["estimate"].iloc[0]
            b0 = boot[:, list(pred.index[(pred.disturbance_type == d) & (pred.years_since_disturbance == a0)])[0]]
            b1 = boot[:, list(pred.index[(pred.disturbance_type == d) & (pred.years_since_disturbance == a1)])[0]]
            diff = b1 - b0
            changes.append({
                "product": product,
                "metric": metric_name,
                "disturbance_type": d,
                "interval": f"{a0}->{a1}",
                "change": p1 - p0,
                "ci95_low": np.nanpercentile(diff, 2.5),
                "ci95_high": np.nanpercentile(diff, 97.5),
            })
    return out, pd.DataFrame(changes), model


def prediction_grid(df: pd.DataFrame, metric_name: str, product: str) -> pd.DataFrame:
    ages = np.linspace(0, 20, 101)
    pred, _, _ = fit_predict(df, metric_name, product, list(ages))
    return pred


def fire_harvest_difference(pred: pd.DataFrame, product: str, metric_name: str) -> pd.DataFrame:
    wide = pred.pivot(index="years_since_disturbance", columns="disturbance_type", values=["estimate", "ci95_low", "ci95_high"])
    rows = []
    for age in pred["years_since_disturbance"].drop_duplicates().sort_values():
        f = pred[(pred.years_since_disturbance == age) & (pred.disturbance_type == "fire_total")].iloc[0]
        h = pred[(pred.years_since_disturbance == age) & (pred.disturbance_type == "harvest_total")].iloc[0]
        rows.append({
            "product": product,
            "metric": metric_name,
            "years_since_disturbance": age,
            "fire_minus_harvest": f["estimate"] - h["estimate"],
            "fire_estimate": f["estimate"],
            "harvest_estimate": h["estimate"],
        })
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame, product: str, metric_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = attach_event_pairs_for_paired_analysis(df)
    wide = df.pivot_table(
        index=["event_pair_id", "disturbance_year", "observation_year", "years_since_disturbance"],
        columns="disturbance_type",
        values="distance",
        aggfunc="median",
    ).reset_index()
    wide = wide.dropna(subset=["fire_total", "harvest_total"]).copy()
    wide["fire_minus_harvest"] = wide["fire_total"] - wide["harvest_total"]
    wide.insert(0, "product", product)
    wide.insert(1, "metric", metric_name)

    model_df = wide[(wide["years_since_disturbance"] >= 0) & (wide["years_since_disturbance"] <= 20)].copy()
    pred = pd.DataFrame({"years_since_disturbance": FIXED_AGES})
    x = spline_features(model_df["years_since_disturbance"].to_numpy())
    model = Ridge(alpha=1e-6, fit_intercept=True).fit(x, model_df["fire_minus_harvest"].to_numpy())
    est = model.predict(spline_features(pred["years_since_disturbance"].to_numpy()))
    pair_ids = model_df["event_pair_id"].dropna().unique()
    boot = []
    for _ in range(N_BOOT):
        sampled = RNG.choice(pair_ids, size=len(pair_ids), replace=True)
        bdf = pd.concat([model_df[model_df["event_pair_id"] == p] for p in sampled], ignore_index=True)
        if bdf["years_since_disturbance"].nunique() < 4:
            continue
        bm = Ridge(alpha=1e-6, fit_intercept=True).fit(
            spline_features(bdf["years_since_disturbance"].to_numpy()),
            bdf["fire_minus_harvest"].to_numpy(),
        )
        boot.append(bm.predict(spline_features(pred["years_since_disturbance"].to_numpy())))
    boot = np.asarray(boot)
    pred.insert(0, "product", product)
    pred.insert(1, "metric", metric_name)
    pred["fire_minus_harvest"] = est
    pred["ci95_low"] = np.nanpercentile(boot, 2.5, axis=0)
    pred["ci95_high"] = np.nanpercentile(boot, 97.5, axis=0)
    support = model_df.groupby("years_since_disturbance").agg(n_pairs=("event_pair_id", "nunique"), n_pair_years=("event_pair_id", "size")).reset_index()
    pred = pred.merge(support, on="years_since_disturbance", how="left")
    return wide, pred


def functional_form(df: pd.DataFrame, product: str, metric_name: str) -> pd.DataFrame:
    rows = []
    model_df = df[(df["years_since_disturbance"] >= 0) & (df["years_since_disturbance"] <= 20)].copy()
    groups = model_df["event_id"].to_numpy()
    n_splits = min(5, pd.Series(groups).nunique())
    for name, design_fn in [("linear", design_linear), ("flexible", design_flexible)]:
        x = design_fn(model_df)
        y = model_df["distance"].to_numpy()
        m = Ridge(alpha=1e-6, fit_intercept=False).fit(x, y)
        pred = m.predict(x)
        rss = float(np.sum((y - pred) ** 2))
        k = x.shape[1]
        aic = len(y) * np.log(rss / len(y)) + 2 * k
        rmses = []
        for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
            cm = Ridge(alpha=1e-6, fit_intercept=False).fit(x[tr], y[tr])
            rmses.append(mean_squared_error(y[te], cm.predict(x[te])) ** 0.5)
        rows.append({
            "product": product,
            "metric": metric_name,
            "model": name,
            "n_event_years": len(model_df),
            "n_events": model_df["event_id"].nunique(),
            "aic_equivalent": aic,
            "event_grouped_cv_rmse": float(np.mean(rmses)),
            "residual_sd": float(np.std(y - pred, ddof=1)),
        })
    return pd.DataFrame(rows)


def within_event_audit(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for product, df in dfs.items():
        for d, g0 in df[(df.years_since_disturbance >= 0) & (df.years_since_disturbance <= 20)].groupby("disturbance_type"):
            slopes = []
            first_last = []
            for event, g in g0.groupby("event_id"):
                if len(g) < 2 or g["years_since_disturbance"].nunique() < 2:
                    continue
                slope = np.polyfit(g["years_since_disturbance"], g["distance"], 1)[0]
                slopes.append(slope)
                gs = g.sort_values("years_since_disturbance")
                first_last.append(gs["distance"].iloc[-1] - gs["distance"].iloc[0])
            rows.append({
                "product": product,
                "disturbance_type": d,
                "n_events": len(slopes),
                "median_within_event_annual_slope": np.nanmedian(slopes),
                "p25_slope": np.nanpercentile(slopes, 25),
                "p75_slope": np.nanpercentile(slopes, 75),
                "median_first_to_last_change": np.nanmedian(first_last),
                "fraction_negative_slopes": float(np.mean(np.asarray(slopes) < 0)) if slopes else np.nan,
            })
    return pd.DataFrame(rows)


def calendar_year_audit(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for product, df in dfs.items():
        for (year, d), g in df.groupby(["observation_year", "disturbance_type"]):
            rows.append({
                "product": product,
                "observation_year": year,
                "disturbance_type": d,
                "n_event_years": len(g),
                "n_events": g["event_id"].nunique(),
                "median_distance": g["distance"].median(),
                "mean_distance": g["distance"].mean(),
                "p25": g["distance"].quantile(0.25),
                "p75": g["distance"].quantile(0.75),
            })
    return pd.DataFrame(rows)


def s2_support_sensitivity(multi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for min_sites in [1, 2, 3]:
        df = prep_product(multi, "sentinel2", "s2_reference_distance", min_s2_sites=min_sites)
        pred, _, _ = fit_predict(df, "s2_reference_distance", "sentinel2", PRIMARY_AGES)
        paired_raw, paired_pred = paired_differences(df, "sentinel2", "s2_reference_distance")
        for _, r in pred.iterrows():
            rows.append({
                "min_valid_sites": min_sites,
                "analysis": "trajectory",
                "disturbance_type": r["disturbance_type"],
                "years_since_disturbance": r["years_since_disturbance"],
                "estimate": r["estimate"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "n_events": r["n_events"],
            })
        for _, r in paired_pred[paired_pred.years_since_disturbance.isin(PRIMARY_AGES)].iterrows():
            rows.append({
                "min_valid_sites": min_sites,
                "analysis": "paired_fire_minus_harvest",
                "disturbance_type": "paired",
                "years_since_disturbance": r["years_since_disturbance"],
                "estimate": r["fire_minus_harvest"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "n_events": r["n_pairs"],
            })
    return pd.DataFrame(rows)


def prebaseline_sensitivity(multi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("sentinel2", "s2_reference_distance", "n_valid_sites_s2"),
        ("alphaearth", "ae_euclidean_distance", "n_valid_sites_ae"),
    ]
    for product, metric, ncol in specs:
        df = multi[["disturbance_type", "event_id", "years_since_disturbance", "observation_year", metric, ncol]].rename(columns={metric: "distance", ncol: "n_valid_sites"}).dropna(subset=["distance"])
        pre = df[df.years_since_disturbance < 0].groupby(["disturbance_type", "event_id"])["distance"].median().rename("pre_distance")
        post = df[df.years_since_disturbance.between(0, 20)].merge(pre, on=["disturbance_type", "event_id"], how="inner")
        post["distance_change"] = post["distance"] - post["pre_distance"]
        for d, g in post.groupby("disturbance_type"):
            for age in PRIMARY_AGES + [20]:
                near = g[g.years_since_disturbance == age]
                if len(near) == 0:
                    continue
                rows.append({
                    "product": product,
                    "disturbance_type": d,
                    "years_since_disturbance": age,
                    "n_events": near["event_id"].nunique(),
                    "median_distance_change": near["distance_change"].median(),
                    "p25": near["distance_change"].quantile(0.25),
                    "p75": near["distance_change"].quantile(0.75),
                })
    return pd.DataFrame(rows)


def cosine_sensitivity(multi: pd.DataFrame) -> pd.DataFrame:
    eu = prep_product(multi, "alphaearth", "ae_euclidean_distance")
    co = prep_product(multi, "alphaearth", "ae_cosine_distance")
    eu_pred, _, _ = fit_predict(eu, "ae_euclidean_distance", "alphaearth", FIXED_AGES)
    co_pred, _, _ = fit_predict(co, "ae_cosine_distance", "alphaearth", FIXED_AGES)
    out = eu_pred[["disturbance_type", "years_since_disturbance", "estimate"]].rename(columns={"estimate": "euclidean_estimate"}).merge(
        co_pred[["disturbance_type", "years_since_disturbance", "estimate"]].rename(columns={"estimate": "cosine_estimate"}),
        on=["disturbance_type", "years_since_disturbance"],
    )
    out["rank_order_note"] = "euclidean and cosine are compared as a technical sensitivity only"
    return out


def landsat_comparison(multi: pd.DataFrame) -> pd.DataFrame:
    cols = ["nbr_gap_median", "nbr_gap_change", "s2_reference_distance", "ae_euclidean_distance"]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pair = multi[[a, b]].dropna()
            rows.append({
                "level": "event_year",
                "metric_a": a,
                "metric_b": b,
                "n": len(pair),
                "pearson": pair[a].corr(pair[b]),
                "spearman": pair[a].corr(pair[b], method="spearman"),
            })
    return pd.DataFrame(rows)


def event_support_by_age(multi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, metric in [("sentinel2", "s2_reference_distance"), ("alphaearth", "ae_euclidean_distance")]:
        df = prep_product(multi, product, metric)
        for age in FIXED_AGES:
            g = df[df.years_since_disturbance == age]
            fire = g[g.disturbance_type == "fire_total"]["event_id"].nunique()
            harv = g[g.disturbance_type == "harvest_total"]["event_id"].nunique()
            paired_source = attach_event_pairs_for_paired_analysis(g)
            paired = paired_source.pivot_table(index=["event_pair_id", "observation_year", "years_since_disturbance"], columns="disturbance_type", values="distance", aggfunc="median").dropna().shape[0]
            rows.append({"product": product, "years_since_disturbance": age, "fire_events": fire, "harvest_events": harv, "complete_pairs": paired})
    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame, product: str, metric_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df[(df.years_since_disturbance >= 0) & (df.years_since_disturbance <= 20)].copy()
    model = Ridge(alpha=1e-6, fit_intercept=False).fit(design_flexible(model_df), model_df.distance.to_numpy())
    model_df["fitted"] = model.predict(design_flexible(model_df))
    model_df["residual"] = model_df["distance"] - model_df["fitted"]
    ev = model_df.groupby("event_id").agg(
        disturbance_type=("disturbance_type", "first"),
        n_event_years=("event_id", "size"),
        mean_abs_residual=("residual", lambda s: float(np.mean(np.abs(s)))),
        max_abs_residual=("residual", lambda s: float(np.max(np.abs(s)))),
    ).reset_index()
    summary = pd.DataFrame([{
        "product": product,
        "metric": metric_name,
        "n_event_years": len(model_df),
        "n_events": model_df.event_id.nunique(),
        "residual_median": model_df.residual.median(),
        "residual_sd": model_df.residual.std(),
        "largest_event_mean_abs_residual": ev.mean_abs_residual.max(),
        "events_above_p95_mean_abs_residual": int((ev.mean_abs_residual >= ev.mean_abs_residual.quantile(0.95)).sum()),
    }])
    model_df["product"] = product
    model_df["metric"] = metric_name
    return summary, model_df


def plot_trajectory(df: pd.DataFrame, grid: pd.DataFrame, product: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {"fire_total": "#b9563b", "harvest_total": "#1f6f5b"}
    labels = {"fire_total": "Wildfire", "harvest_total": "Harvest"}
    for d in ["fire_total", "harvest_total"]:
        obs = df[(df.disturbance_type == d) & df.years_since_disturbance.between(0, 20)]
        ax.scatter(obs.years_since_disturbance, obs.distance, s=12, alpha=0.22, color=colors[d], edgecolor="none")
        g = grid[grid.disturbance_type == d].sort_values("years_since_disturbance")
        ax.plot(g.years_since_disturbance, g.estimate, color=colors[d], lw=2.4, label=labels[d])
        ax.fill_between(g.years_since_disturbance.to_numpy(float), g.ci95_low.to_numpy(float), g.ci95_high.to_numpy(float), color=colors[d], alpha=0.16)
    ax.axvspan(16, 20, color="#777777", alpha=0.08, label="16-20 sensitivity")
    ax.set_xlabel("Years since disturbance")
    ax.set_ylabel(ylabel)
    ax.set_title(product)
    ax.legend(frameon=False)
    ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_difference(diff: pd.DataFrame, ycol: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    d = diff.sort_values("years_since_disturbance")
    ax.axhline(0, color="#222222", lw=1)
    ax.plot(d.years_since_disturbance, d[ycol], color="#263238", lw=2.2, marker="o")
    if {"ci95_low", "ci95_high"}.issubset(d.columns):
        ax.fill_between(d.years_since_disturbance.to_numpy(float), d.ci95_low.to_numpy(float), d.ci95_high.to_numpy(float), color="#607d8b", alpha=0.20)
    ax.axvspan(16, 20, color="#777777", alpha=0.08)
    ax.set_xlabel("Years since disturbance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    multi, phase10_support = load_data()
    s2 = prep_product(multi, "sentinel2", "s2_reference_distance")
    ae = prep_product(multi, "alphaearth", "ae_euclidean_distance")
    dfs = {"sentinel2": s2, "alphaearth": ae}

    s2_pred, s2_changes, _ = fit_predict(s2, "s2_reference_distance", "sentinel2", FIXED_AGES)
    ae_pred, ae_changes, _ = fit_predict(ae, "ae_euclidean_distance", "alphaearth", FIXED_AGES)
    s2_grid = prediction_grid(s2, "s2_reference_distance", "sentinel2")
    ae_grid = prediction_grid(ae, "ae_euclidean_distance", "alphaearth")

    s2_diff = fire_harvest_difference(s2_pred, "sentinel2", "s2_reference_distance")
    ae_diff = fire_harvest_difference(ae_pred, "alphaearth", "ae_euclidean_distance")
    paired_s2, paired_s2_pred = paired_differences(s2, "sentinel2", "s2_reference_distance")
    paired_ae, paired_ae_pred = paired_differences(ae, "alphaearth", "ae_euclidean_distance")

    func = pd.concat([
        functional_form(s2, "sentinel2", "s2_reference_distance"),
        functional_form(ae, "alphaearth", "ae_euclidean_distance"),
    ], ignore_index=True)
    within = within_event_audit(dfs)
    calendar = calendar_year_audit(dfs)
    support_sens = s2_support_sensitivity(multi)
    prebase = prebaseline_sensitivity(multi)
    cosine = cosine_sensitivity(multi)
    landsat = landsat_comparison(multi)
    support = event_support_by_age(multi)
    diag_s2, diag_s2_rows = diagnostics(s2, "sentinel2", "s2_reference_distance")
    diag_ae, diag_ae_rows = diagnostics(ae, "alphaearth", "ae_euclidean_distance")
    diag = pd.concat([diag_s2, diag_ae], ignore_index=True)

    s2_pred.to_csv(OUT_DIR / "s2_fixed_age_predictions.csv", index=False, float_format="%.8f")
    ae_pred.to_csv(OUT_DIR / "ae_fixed_age_predictions.csv", index=False, float_format="%.8f")
    s2_grid.to_csv(OUT_DIR / "s2_model_predictions.csv", index=False, float_format="%.8f")
    ae_grid.to_csv(OUT_DIR / "ae_model_predictions.csv", index=False, float_format="%.8f")
    s2_diff.to_csv(OUT_DIR / "s2_fire_harvest_difference.csv", index=False, float_format="%.8f")
    ae_diff.to_csv(OUT_DIR / "ae_fire_harvest_difference.csv", index=False, float_format="%.8f")
    paired_s2_pred.to_csv(OUT_DIR / "paired_s2_difference.csv", index=False, float_format="%.8f")
    paired_ae_pred.to_csv(OUT_DIR / "paired_ae_difference.csv", index=False, float_format="%.8f")
    func.to_csv(OUT_DIR / "functional_form_comparison.csv", index=False, float_format="%.8f")
    within.to_csv(OUT_DIR / "within_event_change_audit.csv", index=False, float_format="%.8f")
    calendar.to_csv(OUT_DIR / "calendar_year_audit.csv", index=False, float_format="%.8f")
    support_sens.to_csv(OUT_DIR / "s2_site_support_sensitivity.csv", index=False, float_format="%.8f")
    prebase.to_csv(OUT_DIR / "prebaseline_subset_sensitivity.csv", index=False, float_format="%.8f")
    cosine.to_csv(OUT_DIR / "ae_cosine_sensitivity.csv", index=False, float_format="%.8f")
    landsat.to_csv(OUT_DIR / "multivariate_vs_landsat_comparison.csv", index=False, float_format="%.8f")
    support.to_csv(OUT_DIR / "event_support_by_age.csv", index=False, float_format="%.8f")
    diag.to_csv(OUT_DIR / "model_diagnostics_summary.csv", index=False, float_format="%.8f")

    # Add interval changes to model outputs as a compact companion.
    pd.concat([s2_changes, ae_changes], ignore_index=True).to_csv(OUT_DIR / "interval_changes.csv", index=False, float_format="%.8f")
    pd.concat([diag_s2_rows, diag_ae_rows], ignore_index=True).to_csv(OUT_DIR / "model_residual_event_years.csv", index=False, float_format="%.8f")

    plot_trajectory(s2, s2_grid, "Sentinel-2 multiband reference distance", "Standardized spectral distance", FIG_DIR / "s2_reference_distance_trajectory.png")
    plot_trajectory(ae, ae_grid, "AlphaEarth reference distance", "Embedding Euclidean distance", FIG_DIR / "ae_reference_distance_trajectory.png")
    plot_difference(s2_diff, "fire_minus_harvest", "S2 fitted fire - harvest distance", "Distance difference", FIG_DIR / "s2_fire_vs_harvest_difference.png")
    plot_difference(ae_diff, "fire_minus_harvest", "AlphaEarth fitted fire - harvest distance", "Distance difference", FIG_DIR / "ae_fire_vs_harvest_difference.png")
    plot_difference(paired_s2_pred, "fire_minus_harvest", "S2 paired fire - harvest distance", "Paired difference", FIG_DIR / "paired_s2_difference.png")
    plot_difference(paired_ae_pred, "fire_minus_harvest", "AlphaEarth paired fire - harvest distance", "Paired difference", FIG_DIR / "paired_ae_difference.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for product, g in within.groupby("product"):
        ax.scatter(g["median_within_event_annual_slope"], g["fraction_negative_slopes"], s=80, label=product)
        for _, r in g.iterrows():
            ax.text(r["median_within_event_annual_slope"], r["fraction_negative_slopes"], r["disturbance_type"].replace("_total", ""), fontsize=8)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_xlabel("Median within-event annual slope")
    ax.set_ylabel("Fraction of events with negative slope")
    ax.set_title("Within-event direction check")
    ax.legend(frameon=False)
    ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "within_event_change.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for product, marker in [("sentinel2", "o"), ("alphaearth", "s")]:
        g = support[support["product"] == product]
        ax.plot(g.years_since_disturbance, g.complete_pairs, marker=marker, label=f"{product} complete pairs")
    ax.set_xlabel("Years since disturbance")
    ax.set_ylabel("Complete fire-harvest pairs")
    ax.set_title("Event support at fixed ages")
    ax.legend(frameon=False)
    ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_support_by_age.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    x = multi["nbr_gap_median"]
    y = multi["ae_euclidean_distance"]
    mask = x.notna() & y.notna()
    ax.scatter(x[mask], y[mask], s=12, alpha=0.25, color="#435a64", edgecolor="none")
    ax.set_xlabel("Landsat NBR gap")
    ax.set_ylabel("AlphaEarth Euclidean distance")
    ax.set_title("Multivariate vs Landsat/NBR overlap")
    ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "multivariate_vs_landsat.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, rows, title in [(axes[0], diag_s2_rows, "S2 residuals"), (axes[1], diag_ae_rows, "AE residuals")]:
        ax.scatter(rows["fitted"], rows["residual"], s=10, alpha=0.25, color="#455a64", edgecolor="none")
        ax.axhline(0, color="#222222", lw=1)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, color="#dddddd", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_diagnostics.png")
    plt.close(fig)

    summary = {
        "s2_declines_with_age": bool(s2_pred.query("disturbance_type == 'fire_total' and years_since_disturbance == 15")["estimate"].iloc[0] < s2_pred.query("disturbance_type == 'fire_total' and years_since_disturbance == 0")["estimate"].iloc[0]
                                  and s2_pred.query("disturbance_type == 'harvest_total' and years_since_disturbance == 15")["estimate"].iloc[0] < s2_pred.query("disturbance_type == 'harvest_total' and years_since_disturbance == 0")["estimate"].iloc[0]),
        "ae_declines_with_age": bool(ae_pred.query("disturbance_type == 'fire_total' and years_since_disturbance == 15")["estimate"].iloc[0] < ae_pred.query("disturbance_type == 'fire_total' and years_since_disturbance == 0")["estimate"].iloc[0]
                                  and ae_pred.query("disturbance_type == 'harvest_total' and years_since_disturbance == 15")["estimate"].iloc[0] < ae_pred.query("disturbance_type == 'harvest_total' and years_since_disturbance == 0")["estimate"].iloc[0]),
        "alphaearth_euclidean_cosine_note": "Phase 10 and Phase 11 sensitivity show Euclidean and cosine distances are near-redundant; cosine is retained only as a technical check.",
        "primary_age_domain": "0-15 years since disturbance",
        "secondary_age_domain": "16-20 years since disturbance",
        "age_21_plus": "descriptive only; excluded from model fitting",
        "main_limitation": "2017-2025 observation window creates age-period-cohort confounding; older ages are substantially cohort substitution.",
        "functional_form_comparison": func.to_dict(orient="records"),
        "within_event_change_audit": within.to_dict(orient="records"),
        "landsat_comparison": landsat.to_dict(orient="records"),
    }
    (OUT_DIR / "multivariate_recovery_model_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "README.md").write_text(
        "# Phase 11 multivariate recovery model\n\n"
        "This directory contains event-level models of disturbed-reference distance in Sentinel-2 multiband spectral space and AlphaEarth representation space. "
        "The event-year is the inferential unit; site-years are not treated as independent disturbance replicates. "
        "The primary age domain is 0-15 years since disturbance, 16-20 is a cautious sensitivity range, and 21+ is descriptive only.\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
