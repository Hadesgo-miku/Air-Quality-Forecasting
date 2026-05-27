#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
monthly_path = ROOT / "reports/tables/phase_2_monthly_metrics.csv"
out_dir = ROOT / "reports/figures"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(monthly_path)
df = df[df["year_month"].str.startswith("2024-")].copy()
df["month_idx"] = pd.to_datetime(df["year_month"] + "-01")

# focus models (can be expanded)
focus_models = ["ridge_lag", "rf_stable", "lgbm_small", "lgbm_medium", "lgbm_large", "persistence"]
df = df[df["model"].isin(focus_models)].copy()

sns.set_theme(style="whitegrid")

# 1) multi-horizon panels for MAE degradation
horizons = [1, 6, 12, 24, 48, 72]
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()
for ax, h in zip(axes, horizons):
    sub = df[df["horizon_h"] == h].sort_values("month_idx")
    for m, g in sub.groupby("model"):
        ax.plot(g["month_idx"], g["MAE"], marker="o", label=m)
    ax.set_title(f"Monthly MAE Degradation (h={h})")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=45)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / "phase_2_monthly_degradation_all_horizons.png", dpi=150)
plt.close(fig)

# 2) long-horizon focus
long_h = [24, 48, 72]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
for ax, h in zip(axes, long_h):
    sub = df[df["horizon_h"] == h].sort_values("month_idx")
    for m, g in sub.groupby("model"):
        ax.plot(g["month_idx"], g["MAE"], marker="o", label=m)
    ax.set_title(f"Long-horizon MAE drift (h={h})")
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=45)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(out_dir / "phase_2_monthly_degradation_long_horizons.png", dpi=150)
plt.close(fig)

print("Saved:")
print(out_dir / "phase_2_monthly_degradation_all_horizons.png")
print(out_dir / "phase_2_monthly_degradation_long_horizons.png")
