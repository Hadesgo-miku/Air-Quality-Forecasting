# Phase 1 Drift Analysis Report (NYC PM2.5, Single Site)

## 1. 数据范围与站点信息

- 站点：`state=36, county=005, site=0110`
- 污染物：`PM25`
- 固定分析窗口：`2019-01-01 00:00:00` 至 `2024-12-31 23:00:00`
- 频率：小时级

> 注：本轮按项目决策排除 2017–2018，避免将已识别的负值异常密集期引入 Phase 1 主结论。

## 2. 数据覆盖率与质量

（待填充：时间覆盖率、缺失率、负值处理策略执行摘要）

## 3. PM2.5 分布年度变化

（待填充：yearly summary + 年度分布图）

## 4. 季节性和日周期变化

（待填充：year × month heatmap、year × hour profile）

## 5. 分布漂移指标

（待填充：Wasserstein / KS statistic / PSI / mean shift / variance ratio）

## 6. 时间依赖结构变化

（待填充：lag-1/24/168 ACF 指标与对比图）

## 7. 趋势-季节-残差分解

（待填充：STL components + 解释）

## 8. 高污染事件分析

（待填充：固定阈值与分位数阈值事件统计）

## 9. 对后续建模的启示

（待填充：对 Phase 2/3/4 的关键输入结论）

## 10. 下一阶段建议

（待填充）

---

## 附录：本阶段应交付表格

- `reports/tables/yearly_summary.csv`
- `reports/tables/monthly_summary.csv`
- `reports/tables/drift_metrics_by_year.csv`
- `reports/tables/missingness_by_year_month.csv`
- `reports/tables/extreme_events_summary.csv`

## 附录：本阶段应交付图表

- `reports/figures/phase_1_full_series_pm25.png`
- `reports/figures/phase_1_yearly_distribution_boxplot.png`
- `reports/figures/phase_1_yearly_mean_median_p95.png`
- `reports/figures/phase_1_year_month_heatmap.png`
- `reports/figures/phase_1_year_hour_profile.png`
- `reports/figures/phase_1_high_pm_ratio_by_year.png`
- `reports/figures/phase_1_drift_metrics_by_year.png`
- `reports/figures/phase_1_acf_comparison.png`
- `reports/figures/phase_1_stl_components.png`
- `reports/figures/phase_1_missingness_heatmap.png`


## Phase 1 自动运行摘要

```json
{
  "rows": 52608,
  "years": [
    2019,
    2020,
    2021,
    2022,
    2023,
    2024
  ],
  "quantile_thresholds": {
    "0.9": 14.3,
    "0.95": 17.95
  },
  "tables": [
    "yearly_summary.csv",
    "monthly_summary.csv",
    "drift_metrics_by_year.csv",
    "missingness_by_year_month.csv",
    "extreme_events_summary.csv"
  ],
  "figures": [
    "phase_1_full_series_pm25.png",
    "phase_1_yearly_distribution_boxplot.png",
    "phase_1_yearly_mean_median_p95.png",
    "phase_1_year_month_heatmap.png",
    "phase_1_year_hour_profile.png",
    "phase_1_high_pm_ratio_by_year.png",
    "phase_1_drift_metrics_by_year.png",
    "phase_1_acf_comparison.png",
    "phase_1_stl_components.png",
    "phase_1_missingness_heatmap.png"
  ]
}
```
