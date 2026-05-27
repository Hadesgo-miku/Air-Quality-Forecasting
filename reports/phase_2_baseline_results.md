# Phase 2：任务难度与强基线实验结果（最终版）

## 1. 实验目标
本阶段目标是验证：
1. 单站点 PM2.5 任务在不同预测步长下的难度变化；
2. 强基线（persistence / seasonal naive）是否已接近上限；
3. 机器学习模型（Ridge / RF / LGBM）在不同时距上的精度与稳定性；
4. 模型性能是否随测试期月份推进出现退化（degradation）。

---

## 2. 数据与评估设置
- 数据范围：`2019-01-01 00:00:00` ～ `2024-12-31 23:00:00`
- 目标：`PM25`（负值按物理约束截断为 0）
- Horizon：`[1, 6, 12, 24, 48, 72]` 小时
- 特征：`lag + rolling + hour/dow/month`

### 时间切分（无泄露）
- 训练集：`<= 2023-06-30 23:00:00`
- 验证集：`2023-07-01 00:00:00` ～ `2023-12-31 23:00:00`
- 测试集：`2024-01-01 00:00:00` ～ `2024-12-31 23:00:00`

### 模型设置
- Baselines：`persistence` / `seasonal_naive_daily` / `seasonal_naive_weekly` / `rolling_mean_24h` / `rolling_mean_168h`
- ML：`ridge_lag`、`rf_stable`、`lgbm_small/medium/large`
- RF：固定单组偏稳配置（不做多轮堆树搜索）
- LGBM：
  - 先记录完整训练/验证 loss 曲线；
  - 不使用早停直接终止；
  - 用 `best_iter_from_curve + 100` 作为强制截断迭代上限用于最终拟合与预测。

---

## 3. 主要产出文件
### 指标与汇总
- `reports/tables/phase_2_metrics_by_horizon.csv`
- `reports/tables/phase_2_yearly_metrics.csv`
- `reports/tables/phase_2_stability_summary.csv`
- `reports/tables/phase_2_rank_by_horizon.csv`

### 逐时与逐月（本轮重点）
- `reports/tables/phase_2_oos_predictions_full.csv`
- `reports/tables/phase_2_monthly_metrics.csv`

### 图表
- `reports/figures/phase_2_horizon_error_curve.png`
- `reports/figures/phase_2_model_rank_by_horizon.png`
- `reports/figures/phase_2_stability_hmax_mean_std.png`
- `reports/figures/phase_2_monthly_degradation_all_horizons.png`
- `reports/figures/phase_2_monthly_degradation_long_horizons.png`
- `reports/figures/phase_2_lgbm_curve_h*_lgbm_*.png`

---

## 4. 结果总结

### 4.1 按 horizon 的模型表现
- **h=1, h=6：** `persistence` 仍是最强/接近最强。
  - 说明短时预测主要由强自相关与局部惯性主导。
- **h=12,24,48,72：** `ridge_lag` 整体最优或最稳。
  - 且相对 `persistence` 的 MAE 改进随 horizon 增大而增强（长时距更明显）。
- `rf_stable` 在中长时距通常位于第二梯队，表现稳定但未系统超越 ridge。
- `lgbm_small/medium/large` 在短时距可接近 ridge/RF，但在长时距未表现出持续优势。

### 4.2 LGBM 迭代行为（关键诊断）
- 在 `h=1/6/12`，`best_iter_from_curve` 相对较大（几十到两百+），说明有效学习阶段存在。
- 在 `h>=24`，`best_iter_from_curve` 显著变小（常见为 1、2 或几十），说明后续迭代收益迅速减弱。
- 这与“长时距可学习信号衰减 + 跨时段泛化压力上升”一致。

### 4.3 逐月退化（degradation）
基于 `phase_2_monthly_metrics.csv` 与两张 degradation 图可见：
- 各模型在 2024 年存在明显月度波动，非平稳特征显著；
- 长时距（24/48/72h）的月度误差波动幅度明显大于短时距；
- 多模型在年中与秋冬个别月份出现误差抬升，提示固定单指标会掩盖真实部署风险。

---

## 5. 结论（Phase 2）
1. 该任务并非“整体简单任务”：
   - 短时距由强基线主导；
   - 长时距存在可学习空间，但对模型稳健性要求高。
2. 在当前特征与切分下，`ridge_lag` 提供了最稳定、性价比最高的中长时距结果。
3. GBDT（LGBM）并非无效，但在长时距上有效迭代区间很短，盲目增加迭代会造成训练浪费。
4. 逐月评估显示时间后段风险与波动明显，支持下一阶段必须进入部署式评估。

---

## 6. 下一阶段建议（Phase 3）
建议立即进入 **walk-forward 部署风险评估**：
- expanding-window 与 rolling-window 对比；
- 关注“最差年份/最差月份”而非仅平均 MAE；
- 检验模型排名是否随时间发生结构性变化；
- 联动 Phase 1 漂移指标，分析误差退化是否与 drift 同步增强。
