#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


@dataclass
class Cfg:
    input_path: Path
    report_path: Path
    tables_dir: Path
    figures_dir: Path
    start: pd.Timestamp
    end: pd.Timestamp
    target_col: str
    horizons: list[int]
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_end: pd.Timestamp
    lags: list[int]
    rolling_windows: list[int]
    rf_param_sets: list[dict]
    lgbm_param_sets: list[dict]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def load_cfg(path: Path) -> Cfg:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    exp = d["experiment"]
    return Cfg(
        input_path=Path(d["input"]["wide_dataset"]),
        report_path=Path(d["outputs"]["report_markdown"]),
        tables_dir=Path(d["outputs"]["tables_dir"]),
        figures_dir=Path(d["outputs"]["figures_dir"]),
        start=pd.Timestamp(d["study_scope"]["temporal_window"]["start"]),
        end=pd.Timestamp(d["study_scope"]["temporal_window"]["end"]),
        target_col=exp["target_col"],
        horizons=[int(x) for x in exp["horizons"]],
        train_end=pd.Timestamp(exp["train_end"]),
        valid_end=pd.Timestamp(exp["valid_end"]),
        test_end=pd.Timestamp(exp["test_end"]),
        lags=[int(x) for x in exp["lags"]],
        rolling_windows=[int(x) for x in exp["rolling_windows"]],
        rf_param_sets=exp.get("rf_param_sets", []),
        lgbm_param_sets=exp.get("lgbm_param_sets", []),
    )


def prep_df(cfg: Cfg) -> pd.DataFrame:
    log(f"Loading parquet: {cfg.input_path}")
    df = pd.read_parquet(cfg.input_path)
    df["dt_hour"] = pd.to_datetime(df["dt_hour"])
    df = df.sort_values("dt_hour")
    df = df[(df["dt_hour"] >= cfg.start) & (df["dt_hour"] <= cfg.end)].copy()

    y = pd.to_numeric(df[cfg.target_col], errors="coerce")
    y[y < 0] = 0.0
    df[cfg.target_col] = y

    df["year"] = df["dt_hour"].dt.year
    df["hour"] = df["dt_hour"].dt.hour
    df["dow"] = df["dt_hour"].dt.dayofweek
    df["month"] = df["dt_hour"].dt.month

    log(f"Prepared rows: {len(df):,}, range: {df['dt_hour'].min()} -> {df['dt_hour'].max()}")
    return df


def add_features(df: pd.DataFrame, target_col: str, lags: list[int], roll_ws: list[int], h: int) -> pd.DataFrame:
    out = df[["dt_hour", target_col, "year", "hour", "dow", "month"]].copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for w in roll_ws:
        out[f"roll_mean_{w}"] = out[target_col].shift(1).rolling(w, min_periods=max(2, w // 4)).mean()
        out[f"roll_std_{w}"] = out[target_col].shift(1).rolling(w, min_periods=max(2, w // 4)).std()
    out["target"] = out[target_col].shift(-h)
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if y_t.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "N": 0}
    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "R2": float(r2_score(y_t, y_p)),
        "N": int(y_t.size),
    }


def run(cfg: Cfg) -> None:
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)

    df = prep_df(cfg)

    all_metrics: list[dict] = []
    yearly_metrics: list[dict] = []
    oos_full_records: list[dict] = []

    total_model_runs = len(cfg.horizons) * (7 + len(cfg.rf_param_sets) + len(cfg.lgbm_param_sets))
    done = 0
    log(f"Planned model runs: {total_model_runs}")

    for h in cfg.horizons:
        log(f"=== Horizon {h}h: building features ===")
        work = add_features(df, cfg.target_col, cfg.lags, cfg.rolling_windows, h)

        test_m = (work["dt_hour"] > cfg.valid_end) & (work["dt_hour"] <= cfg.test_end)
        y_test = work.loc[test_m, "target"].values
        idx_test = work.loc[test_m, "dt_hour"]

        preds = {}
        preds["persistence"] = work.loc[test_m, cfg.target_col].values
        preds["seasonal_naive_daily"] = work.loc[test_m, "lag_24"].values if "lag_24" in work.columns else np.full(y_test.shape, np.nan)
        preds["seasonal_naive_weekly"] = work.loc[test_m, "lag_168"].values if "lag_168" in work.columns else np.full(y_test.shape, np.nan)
        preds["rolling_mean_24h"] = work.loc[test_m, "roll_mean_24"].values if "roll_mean_24" in work.columns else np.full(y_test.shape, np.nan)
        preds["rolling_mean_168h"] = work.loc[test_m, "roll_mean_168"].values if "roll_mean_168" in work.columns else np.full(y_test.shape, np.nan)

        feature_cols = [c for c in work.columns if c.startswith("lag_") or c.startswith("roll_")] + ["hour", "dow", "month"]
        ml = work.dropna(subset=feature_cols + ["target"]).copy()
        tr_fit = ml["dt_hour"] <= cfg.train_end
        va_fit = (ml["dt_hour"] > cfg.train_end) & (ml["dt_hour"] <= cfg.valid_end)
        te = (ml["dt_hour"] > cfg.valid_end) & (ml["dt_hour"] <= cfg.test_end)
        X_tr = ml.loc[tr_fit, feature_cols]
        y_tr = ml.loc[tr_fit, "target"]
        X_va = ml.loc[va_fit, feature_cols]
        y_va = ml.loc[va_fit, "target"]
        X_te = ml.loc[te, feature_cols]

        if len(X_tr) > 0 and len(X_te) > 0:
            log(f"H={h}: train rows={len(X_tr):,}, valid rows={len(X_va):,}, test rows={len(X_te):,}, features={len(feature_cols)}")

            # ridge
            t0 = time.time()
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_tr, y_tr)
            pred_map = pd.Series(ridge.predict(X_te), index=ml.loc[te, "dt_hour"]).to_dict()
            preds["ridge_lag"] = np.array([pred_map.get(ts, np.nan) for ts in idx_test])
            done += 1
            log(f"[{done}/{total_model_runs}] ridge_lag done in {time.time()-t0:.2f}s")

            # RF parameter sets
            for i, p in enumerate(cfg.rf_param_sets, start=1):
                name = p.get("name", f"rf_set{i}")
                kwargs = {k: v for k, v in p.items() if k != "name"}
                kwargs.setdefault("random_state", 42)
                t0 = time.time()
                rf = RandomForestRegressor(**kwargs)
                rf.fit(X_tr, y_tr)
                pred_map = pd.Series(rf.predict(X_te), index=ml.loc[te, "dt_hour"]).to_dict()
                preds[name] = np.array([pred_map.get(ts, np.nan) for ts in idx_test])
                done += 1
                log(f"[{done}/{total_model_runs}] {name} done in {time.time()-t0:.2f}s params={json.dumps(kwargs)}")

            # LGBM parameter sets (no early stopping; record train/valid loss curves)
            for i, p in enumerate(cfg.lgbm_param_sets, start=1):
                name = p.get("name", f"lgbm_set{i}")
                if HAS_LGBM:
                    from lightgbm import record_evaluation
                    kwargs = {k: v for k, v in p.items() if k not in {"name", "early_stopping_rounds"}}
                    kwargs.setdefault("random_state", 42)
                    t0 = time.time()
                    model = LGBMRegressor(**kwargs)

                    eval_result = {}
                    if len(X_va) > 0:
                        model.fit(
                            X_tr,
                            y_tr,
                            eval_set=[(X_tr, y_tr), (X_va, y_va)],
                            eval_names=["train", "valid"],
                            eval_metric="l1",
                            callbacks=[record_evaluation(eval_result)],
                        )
                    else:
                        model.fit(
                            X_tr,
                            y_tr,
                            eval_set=[(X_tr, y_tr)],
                            eval_names=["train"],
                            eval_metric="l1",
                            callbacks=[record_evaluation(eval_result)],
                        )

                    pred_map = pd.Series(model.predict(X_te), index=ml.loc[te, "dt_hour"]).to_dict()
                    preds[name] = np.array([pred_map.get(ts, np.nan) for ts in idx_test])

                    train_curve = eval_result.get("train", {}).get("l1", [])
                    valid_curve = eval_result.get("valid", {}).get("l1", [])

                    if len(valid_curve) > 0:
                        best_iter_from_curve = int(np.argmin(valid_curve) + 1)
                        forced_stop_iter = min(int(kwargs.get("n_estimators", len(train_curve))), best_iter_from_curve + 100)
                    else:
                        best_iter_from_curve = None
                        forced_stop_iter = int(kwargs.get("n_estimators", len(train_curve)))

                    if forced_stop_iter < int(kwargs.get("n_estimators", forced_stop_iter)):
                        model = LGBMRegressor(**{**kwargs, "n_estimators": forced_stop_iter})
                        model.fit(X_tr, y_tr)

                    curve_df = pd.DataFrame({
                        "iter": np.arange(1, len(train_curve) + 1),
                        "train_l1": train_curve,
                        "valid_l1": valid_curve if len(valid_curve) == len(train_curve) else [np.nan] * len(train_curve),
                        "horizon_h": h,
                        "model": name,
                        "best_iter_from_curve": best_iter_from_curve,
                        "forced_stop_iter": forced_stop_iter,
                    })
                    curve_path = cfg.tables_dir / f"phase_2_lgbm_curve_h{h}_{name}.csv"
                    curve_df.to_csv(curve_path, index=False)

                    if len(curve_df) > 0:
                        fig, ax = plt.subplots(figsize=(9, 4.5))
                        ax.plot(curve_df["iter"], curve_df["train_l1"], label="train_l1")
                        if curve_df["valid_l1"].notna().any():
                            ax.plot(curve_df["iter"], curve_df["valid_l1"], label="valid_l1")
                        ax.set_title(f"LGBM loss curve: h={h}, model={name}")
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("L1 loss")
                        ax.legend()
                        fig.tight_layout()
                        fig.savefig(cfg.figures_dir / f"phase_2_lgbm_curve_h{h}_{name}.png", dpi=140)
                        plt.close(fig)

                    done += 1
                    log(f"[{done}/{total_model_runs}] {name} done in {time.time()-t0:.2f}s (no early stopping, best_iter_from_curve={best_iter_from_curve}, forced_stop_iter={forced_stop_iter}, logged_iters={len(train_curve)}) params={json.dumps(kwargs)}")
                else:
                    preds[name] = np.full(y_test.shape, np.nan)
                    done += 1
                    log(f"[{done}/{total_model_runs}] {name} skipped (lightgbm not installed)")
        else:
            preds["ridge_lag"] = np.full(y_test.shape, np.nan)

        base_mae = metrics(y_test, preds["persistence"])["MAE"]
        base2_mae = metrics(y_test, preds["seasonal_naive_daily"])["MAE"]

        for model, pred in preds.items():
            m = metrics(y_test, pred)
            m.update({
                "horizon_h": h,
                "model": model,
                "improve_vs_persistence_mae_pct": float((base_mae - m["MAE"]) / base_mae * 100) if np.isfinite(base_mae) and base_mae > 0 else np.nan,
                "improve_vs_daily_naive_mae_pct": float((base2_mae - m["MAE"]) / base2_mae * 100) if np.isfinite(base2_mae) and base2_mae > 0 else np.nan,
            })
            all_metrics.append(m)

            tdf = pd.DataFrame({
                "dt_hour": idx_test.values,
                "year": work.loc[test_m, "year"].values,
                "y_true": y_test,
                "y_pred": pred,
            }).dropna(subset=["y_true", "y_pred"])

            if not tdf.empty:
                tdf2 = tdf.copy()
                tdf2["year_month"] = pd.to_datetime(tdf2["dt_hour"]).dt.to_period("M").astype(str)
                for _, rr in tdf2.iterrows():
                    oos_full_records.append({
                        "dt_hour": rr["dt_hour"],
                        "year_month": rr["year_month"],
                        "horizon_h": h,
                        "model": model,
                        "y_true": rr["y_true"],
                        "y_pred": rr["y_pred"],
                    })

            for y, g in tdf.groupby("year"):
                mm = metrics(g["y_true"].values, g["y_pred"].values)
                yearly_metrics.append({"horizon_h": h, "model": model, "year": int(y), **mm})

    metrics_df = pd.DataFrame(all_metrics).sort_values(["horizon_h", "MAE", "RMSE"])
    yearly_df = pd.DataFrame(yearly_metrics).sort_values(["horizon_h", "model", "year"])

    # stability summary by (horizon, model): mean/std/cv/worst-best gap over years
    if not yearly_df.empty:
        g = yearly_df.groupby(["horizon_h", "model"]) ["MAE"]
        stab = g.agg(mae_mean="mean", mae_std="std", mae_min="min", mae_max="max").reset_index()
        stab["mae_cv"] = stab["mae_std"] / stab["mae_mean"]
        stab["mae_range"] = stab["mae_max"] - stab["mae_min"]
        stability_records = stab.to_dict("records")
        stability_df = stab.sort_values(["horizon_h", "mae_mean"])
    else:
        stability_df = pd.DataFrame(columns=["horizon_h", "model", "mae_mean", "mae_std", "mae_min", "mae_max", "mae_cv", "mae_range"])

    # ranks per horizon by MAE
    rank_df = metrics_df.copy()
    rank_df["rank_mae"] = rank_df.groupby("horizon_h")["MAE"].rank(method="dense")

    metrics_df.to_csv(cfg.tables_dir / "phase_2_metrics_by_horizon.csv", index=False)
    yearly_df.to_csv(cfg.tables_dir / "phase_2_yearly_metrics.csv", index=False)
    stability_df.to_csv(cfg.tables_dir / "phase_2_stability_summary.csv", index=False)
    rank_df.sort_values(["horizon_h", "rank_mae", "MAE"]).to_csv(cfg.tables_dir / "phase_2_rank_by_horizon.csv", index=False)

    oos_full_df = pd.DataFrame(oos_full_records)
    if not oos_full_df.empty:
        oos_full_df = oos_full_df.sort_values(["model", "horizon_h", "dt_hour"])
        oos_full_df.to_csv(cfg.tables_dir / "phase_2_oos_predictions_full.csv", index=False)

        monthly_rows = []
        for (model, h, ym), g in oos_full_df.groupby(["model", "horizon_h", "year_month"]):
            mm = metrics(g["y_true"].values, g["y_pred"].values)
            monthly_rows.append({"model": model, "horizon_h": h, "year_month": ym, **mm})
        monthly_df = pd.DataFrame(monthly_rows).sort_values(["model", "horizon_h", "year_month"])
        monthly_df.to_csv(cfg.tables_dir / "phase_2_monthly_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    for model, gg in metrics_df.groupby("model"):
        ax.plot(gg["horizon_h"], gg["MAE"], marker="o", label=model)
    ax.set_title("Phase 2 Horizon-Error Curve (MAE)")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "phase_2_horizon_error_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for model, gg in rank_df.groupby("model"):
        ax.plot(gg["horizon_h"], gg["rank_mae"], marker="o", label=model)
    ax.invert_yaxis()
    ax.set_title("Model Rank by Horizon (lower rank is better)")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Rank by MAE")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "phase_2_model_rank_by_horizon.png", dpi=150)
    plt.close(fig)

    if not stability_df.empty:
        h_focus = max(cfg.horizons)
        plot_df = stability_df[stability_df["horizon_h"] == h_focus].sort_values("mae_mean")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(plot_df["model"], plot_df["mae_mean"], yerr=plot_df["mae_std"].fillna(0), capsize=4)
        ax.set_title(f"Stability on MAE across years (h={h_focus}h): mean ± std")
        ax.set_ylabel("MAE")
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(cfg.figures_dir / "phase_2_stability_hmax_mean_std.png", dpi=150)
        plt.close(fig)

    best = metrics_df.loc[metrics_df.groupby("horizon_h")["MAE"].idxmin()].sort_values("horizon_h")
    top_stab_txt = stability_df.sort_values(["horizon_h", "mae_mean"]).groupby("horizon_h").head(3)

    report = f"""# Phase 2：任务难度与强基线实验结果（含 RF/LGBM 参数组）

## 1. 训练/测试形式
- 固定时间切分（非 walk-forward）
- 训练（用于拟合）: <= {cfg.valid_end}
- 测试: ({cfg.valid_end}, {cfg.test_end}]
- 说明：本版本为你要求的“轻量参数组比较”，不做重型调参。

## 2. 参数组设置
- RF 参数组数量：{len(cfg.rf_param_sets)}
- LGBM 参数组数量：{len(cfg.lgbm_param_sets)}

## 3. 输出文件
- `reports/tables/phase_2_metrics_by_horizon.csv`
- `reports/tables/phase_2_yearly_metrics.csv`
- `reports/tables/phase_2_stability_summary.csv`
- `reports/tables/phase_2_rank_by_horizon.csv`
- `reports/figures/phase_2_horizon_error_curve.png`
- `reports/figures/phase_2_model_rank_by_horizon.png`
- `reports/figures/phase_2_stability_hmax_mean_std.png`

## 4. 每个 horizon 最优模型（按 MAE）
{best.to_markdown(index=False)}

## 5. 稳定性（不是只看平均）
稳定性指标按“同一 horizon 的跨年份 MAE”统计：
- `mae_std`（越小越稳）
- `mae_cv = mae_std / mae_mean`（相对波动，越小越稳）
- `mae_range = max - min`（最差-最好跨度，越小越稳）

各 horizon 前 3（按 mae_mean）如下：
{top_stab_txt.to_markdown(index=False) if not top_stab_txt.empty else '暂无可用稳定性统计'}

## 6. 备注
- 若 LightGBM 未安装，对应模型会被跳过并记为 NaN。
- 下一步建议：在 Phase 3 使用 walk-forward 复核“精度 + 稳定性”是否保持。
"""
    cfg.report_path.write_text(report, encoding="utf-8")

    log("All outputs written successfully.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    cfg = load_cfg(Path(args.config))
    run(cfg)
    print("Phase 2 baseline experiment done.")


if __name__ == "__main__":
    main()
