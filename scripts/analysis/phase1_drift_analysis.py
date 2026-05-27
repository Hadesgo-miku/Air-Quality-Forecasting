#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL


@dataclass
class Cfg:
    input_path: Path
    report_path: Path
    tables_dir: Path
    figures_dir: Path
    start: pd.Timestamp
    end: pd.Timestamp
    ref_years: list[int]
    fixed_threshold: float
    q_thresholds: list[float]


def setup_fonts_for_mac() -> None:
    # macOS-friendly Chinese font fallback
    font_candidates = [
        "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"
    ]
    plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["axes.unicode_minus"] = False


def load_cfg(path: Path) -> Cfg:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cfg(
        input_path=Path(d["input"]["wide_dataset"]),
        report_path=Path(d["outputs"]["report_markdown"]),
        tables_dir=Path(d["outputs"]["tables_dir"]),
        figures_dir=Path(d["outputs"]["figures_dir"]),
        start=pd.Timestamp(d["study_scope"]["temporal_window"]["start"]),
        end=pd.Timestamp(d["study_scope"]["temporal_window"]["end"]),
        ref_years=[int(x) for x in d["reference_period_for_drift"]["years"]],
        fixed_threshold=float(d["thresholds"]["fixed"][0]),
        q_thresholds=[float(x) for x in d["thresholds"]["quantile_based"]],
    )


def psi_score(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    eps = 1e-12
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(ref, qs)
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    r_hist, _ = np.histogram(ref, bins=cuts)
    c_hist, _ = np.histogram(cur, bins=cuts)
    r = np.clip(r_hist / max(r_hist.sum(), 1), eps, None)
    c = np.clip(c_hist / max(c_hist.sum(), 1), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def run(cfg: Cfg) -> dict:
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.input_path)
    df["dt_hour"] = pd.to_datetime(df["dt_hour"])
    df = df.sort_values("dt_hour")
    df = df[(df["dt_hour"] >= cfg.start) & (df["dt_hour"] <= cfg.end)].copy()

    # PM2.5 physical constraint
    df["PM25"] = pd.to_numeric(df["PM25"], errors="coerce")
    df.loc[df["PM25"] < 0, "PM25"] = 0.0

    df["year"] = df["dt_hour"].dt.year
    df["month"] = df["dt_hour"].dt.month
    df["hour"] = df["dt_hour"].dt.hour
    df["dow"] = df["dt_hour"].dt.dayofweek
    df["season"] = df["month"].map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})

    all_q = {q: float(df["PM25"].quantile(q)) for q in cfg.q_thresholds}

    # Table 1 yearly_summary.csv
    rows = []
    for y, g in df.groupby("year"):
        pm = g["PM25"].dropna()
        if len(pm) == 0:
            continue
        rows.append({
            "year": int(y), "count": int(pm.size), "missing_rate": float(g["PM25"].isna().mean()),
            "mean": pm.mean(), "median": pm.median(), "std": pm.std(), "min": pm.min(), "max": pm.max(),
            "p05": pm.quantile(0.05), "p25": pm.quantile(0.25), "p75": pm.quantile(0.75),
            "p95": pm.quantile(0.95), "p99": pm.quantile(0.99), "iqr": pm.quantile(0.75)-pm.quantile(0.25),
            "skewness": skew(pm, bias=False), "kurtosis": kurtosis(pm, bias=False),
            "exceed_35_count": int((pm > cfg.fixed_threshold).sum()),
            "exceed_35_ratio": float((pm > cfg.fixed_threshold).mean()),
            "exceed_p90_count": int((pm > all_q.get(0.90, np.nan)).sum()) if 0.90 in all_q else np.nan,
            "exceed_p90_ratio": float((pm > all_q.get(0.90, np.nan)).mean()) if 0.90 in all_q else np.nan,
            "exceed_p95_count": int((pm > all_q.get(0.95, np.nan)).sum()) if 0.95 in all_q else np.nan,
            "exceed_p95_ratio": float((pm > all_q.get(0.95, np.nan)).mean()) if 0.95 in all_q else np.nan,
        })
    yearly = pd.DataFrame(rows).sort_values("year")
    yearly.to_csv(cfg.tables_dir / "yearly_summary.csv", index=False)

    # Table 2 monthly_summary.csv
    monthly = df.groupby(["year", "month"], as_index=False).agg(
        mean_pm25=("PM25", "mean"), median_pm25=("PM25", "median"), p95_pm25=("PM25", lambda s: s.quantile(0.95)), count=("PM25", "size")
    )
    monthly.to_csv(cfg.tables_dir / "monthly_summary.csv", index=False)

    # Table 3 drift_metrics_by_year.csv
    ref = df[df["year"].isin(cfg.ref_years)]["PM25"].dropna().values
    drift = []
    for y, g in df.groupby("year"):
        cur = g["PM25"].dropna().values
        if len(cur) == 0 or len(ref) == 0:
            continue
        ks = ks_2samp(ref, cur)
        drift.append({
            "year": int(y),
            "wasserstein": float(wasserstein_distance(ref, cur)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "psi": psi_score(ref, cur, bins=10),
            "mean_shift": float(np.mean(cur) - np.mean(ref)),
            "variance_ratio": float(np.var(cur) / np.var(ref)) if np.var(ref) > 0 else np.nan,
            "q50_shift": float(np.quantile(cur, 0.5) - np.quantile(ref, 0.5)),
            "q95_shift": float(np.quantile(cur, 0.95) - np.quantile(ref, 0.95)),
        })
    drift_df = pd.DataFrame(drift).sort_values("year")
    drift_df.to_csv(cfg.tables_dir / "drift_metrics_by_year.csv", index=False)

    # Table 4 missingness_by_year_month.csv
    miss = df.assign(is_missing=df["PM25"].isna()).groupby(["year", "month"], as_index=False).agg(
        missing_rate=("is_missing", "mean"), rows=("is_missing", "size")
    )
    miss.to_csv(cfg.tables_dir / "missingness_by_year_month.csv", index=False)

    # Table 5 extreme_events_summary.csv
    th = cfg.fixed_threshold
    events = []
    for y, g in df.groupby("year"):
        s = g.sort_values("dt_hour")["PM25"].fillna(-np.inf).values > th
        # episodes
        starts = np.where((s[1:] & ~s[:-1]))[0] + 1
        if s.size and s[0]:
            starts = np.r_[0, starts]
        ends = np.where((~s[1:] & s[:-1]))[0]
        if s.size and s[-1]:
            ends = np.r_[ends, s.size - 1]
        lens = (ends - starts + 1) if len(starts) else np.array([])
        events.append({
            "year": int(y),
            "high_hours": int(s.sum()),
            "high_ratio": float(s.mean()),
            "episode_count": int(len(lens)),
            "episode_mean_duration_hours": float(lens.mean()) if len(lens) else 0.0,
            "episode_max_duration_hours": int(lens.max()) if len(lens) else 0,
        })
    events_df = pd.DataFrame(events).sort_values("year")
    events_df.to_csv(cfg.tables_dir / "extreme_events_summary.csv", index=False)

    # Figures
    setup_fonts_for_mac()
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["dt_hour"], df["PM25"], linewidth=0.6)
    ax.set_title("PM2.5 Full Series (2019-2024)")
    ax.set_ylabel("μg/m^3")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_full_series_pm25.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="year", y="PM25", ax=ax, showfliers=False)
    ax.set_title("Yearly PM2.5 Distribution")
    ax.set_ylabel("μg/m^3")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_yearly_distribution_boxplot.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["year"], yearly["mean"], marker="o", label="mean")
    ax.plot(yearly["year"], yearly["median"], marker="o", label="median")
    ax.plot(yearly["year"], yearly["p95"], marker="o", label="p95")
    ax.legend(); ax.set_title("Yearly Mean/Median/P95"); ax.set_ylabel("μg/m^3")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_yearly_mean_median_p95.png", dpi=150); plt.close(fig)

    ym = monthly.pivot(index="year", columns="month", values="mean_pm25")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(ym, cmap="YlOrRd", ax=ax)
    ax.set_title("Year × Month Mean PM2.5")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_year_month_heatmap.png", dpi=150); plt.close(fig)

    yh = df.groupby(["year", "hour"], as_index=False)["PM25"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for y in sorted(yh["year"].unique()):
        p = yh[yh["year"] == y]
        ax.plot(p["hour"], p["PM25"], label=str(y))
    ax.set_title("Year × Hour-of-day Mean PM2.5")
    ax.set_xlabel("hour"); ax.set_ylabel("μg/m^3"); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_year_hour_profile.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(events_df["year"].astype(str), events_df["high_ratio"])
    ax.set_title(f"High PM2.5 Ratio by Year (>{th})")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_high_pm_ratio_by_year.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(drift_df["year"], drift_df["wasserstein"], marker="o", label="wasserstein")
    ax.plot(drift_df["year"], drift_df["psi"], marker="o", label="psi")
    ax.plot(drift_df["year"], drift_df["ks_statistic"], marker="o", label="ks")
    ax.legend(); ax.set_title("Drift Metrics by Year")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_drift_metrics_by_year.png", dpi=150); plt.close(fig)

    years_show = sorted(df["year"].unique())[:1] + [sorted(df["year"].unique())[len(sorted(df["year"].unique()))//2]] + sorted(df["year"].unique())[-1:]
    fig, ax = plt.subplots(figsize=(10, 5))
    for y in years_show:
        s = df[df["year"] == y]["PM25"].dropna().values
        ac = acf(s, nlags=168, fft=True)
        ax.plot(ac, label=str(y))
    ax.set_title("ACF comparison (selected years)")
    ax.set_xlabel("lag(hour)")
    ax.legend()
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_acf_comparison.png", dpi=150); plt.close(fig)

    stl_series = df.set_index("dt_hour")["PM25"].interpolate(limit_direction="both")
    stl = STL(stl_series, period=24, robust=True).fit()
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(stl.trend, linewidth=0.8); axes[0].set_title("STL Trend")
    axes[1].plot(stl.seasonal, linewidth=0.8); axes[1].set_title("STL Seasonal")
    axes[2].plot(stl.resid, linewidth=0.8); axes[2].set_title("STL Residual")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_stl_components.png", dpi=150); plt.close(fig)

    miss_heat = miss.pivot(index="year", columns="month", values="missing_rate")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(miss_heat, cmap="Blues", vmin=0, vmax=max(0.01, float(miss_heat.max().max() if not miss_heat.empty else 0.01)), ax=ax)
    ax.set_title("Missingness Heatmap")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "phase_1_missingness_heatmap.png", dpi=150); plt.close(fig)

    return {
        "rows": int(len(df)),
        "years": sorted(df["year"].unique().tolist()),
        "quantile_thresholds": all_q,
        "tables": [
            "yearly_summary.csv", "monthly_summary.csv", "drift_metrics_by_year.csv",
            "missingness_by_year_month.csv", "extreme_events_summary.csv"
        ],
        "figures": [
            "phase_1_full_series_pm25.png", "phase_1_yearly_distribution_boxplot.png", "phase_1_yearly_mean_median_p95.png",
            "phase_1_year_month_heatmap.png", "phase_1_year_hour_profile.png", "phase_1_high_pm_ratio_by_year.png",
            "phase_1_drift_metrics_by_year.png", "phase_1_acf_comparison.png", "phase_1_stl_components.png", "phase_1_missingness_heatmap.png"
        ]
    }


def write_report_append(cfg: Cfg, meta: dict) -> None:
    txt = cfg.report_path.read_text(encoding="utf-8") if cfg.report_path.exists() else ""
    appendix = "\n\n## Phase 1 自动运行摘要\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```\n"
    cfg.report_path.write_text(txt + appendix, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    meta = run(cfg)
    write_report_append(cfg, meta)

    msg = f"Phase 1 done. rows={meta['rows']}, years={meta['years']}"
    print(msg)
    if args.log_file:
        p = Path(args.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
