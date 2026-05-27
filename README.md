# Air-Quality-Forecasting（单站点 PM2.5：阶段性实验仓库）

本仓库当前聚焦 **New York 单站点 PM2.5 小时级预测**，核心目标是先回答“这个问题值不值得继续做、难点在哪里”，而不是直接追求复杂模型。

目前已完成：
- Phase 1：分布漂移统计分析；
- Phase 2：任务难度与强基线对比（含逐月退化分析）。

---

## 当前结论（简要）

1. **短期预测（1h / 6h）被强基线主导**：
   `persistence` 非常强，说明短时序列惯性显著。

2. **中长时距（12h~72h）并非无解，但更考验稳定性**：
   当前设置下 `ridge_lag` 往往更稳，`RF` 次之；`LGBM` 在长时距上的迭代收益明显变弱。

3. **逐月表现存在明显波动**：
   同一年内（月度）误差并不平稳，提示“只看整体平均 MAE 会掩盖部署风险”。

4. **下一步必须做部署式评估（Phase 3）**：
   需要 walk-forward / retraining 来验证模型在时间后段的泛化稳定性。

---

## 已产出报告与结果

### Phase 1（漂移分析）
- `reports/phase_1_drift_analysis.md`
- `reports/phase_1_drift_analysis_explained.md`
- `reports/tables/yearly_summary.csv`
- `reports/tables/monthly_summary.csv`
- `reports/tables/drift_metrics_by_year.csv`

### Phase 2（基线实验）
- `reports/phase_2_baseline_results.md`
- `reports/tables/phase_2_metrics_by_horizon.csv`
- `reports/tables/phase_2_yearly_metrics.csv`
- `reports/tables/phase_2_stability_summary.csv`
- `reports/tables/phase_2_rank_by_horizon.csv`

### Phase 2（逐月退化分析）
- `reports/tables/phase_2_oos_predictions_full.csv`
- `reports/tables/phase_2_monthly_metrics.csv`
- `reports/figures/phase_2_monthly_degradation_all_horizons.png`
- `reports/figures/phase_2_monthly_degradation_long_horizons.png`

---

## 复现命令

### 1) Phase 1 漂移分析
```bash
python3 scripts/analysis/phase1_drift_analysis.py --config configs/phase_1_drift.yaml
```

### 2) Phase 2 基线与稳定性
```bash
python3 scripts/analysis/phase2_baselines.py --config configs/phase_2_baselines.yaml
```

### 3) 绘制逐月退化图
```bash
python3 scripts/analysis/plot_phase2_monthly_degradation.py
```

---

## 后续计划（简述）

- **Phase 3**：walk-forward 部署风险评估（固定切分 vs expanding/rolling）
- **Phase 4**：定期重训策略与窗口长度
- **Phase 5**：高污染事件预测与漏报/误报分析
- **Phase 6**：残差诊断、分解与可解释性

更完整的分期规范与协作说明见：
- `实验分期与协作方案.md`
