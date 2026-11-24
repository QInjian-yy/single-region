# 第四周实验

本周完成 6 组 CAMELYON16 Seed123 五折消融，共 30 折。每组分别从 K=4 NoEMA 基线出发，比较一处改动。此前的结果见[第三周实验](../week3_experiments/README.md)。

## 测试集指标

Test AUC 为五折等权平均 ± 样本标准差。Δ AUC 为相对基线的百分点变化。ACC、BACC、肿瘤召回率和特异度为五折等权平均（%），各折沿用验证集选出的判定阈值。

| 实验与汇总记录 | Test AUC | Δ AUC（百分点） | ACC | BACC | 肿瘤召回率 | 特异度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [K=4 NoEMA 基线](../week3_experiments/s123_hrk4_noema_5fold_20261119_092239/summary.json) | 0.7719 ± 0.0661 | — | 67.95 | 66.55 | 58.75 | 74.36 |
| [Region 内 HR 自注意力](s123_hrk4_noema_hrselfattn_h4_gate0_5fold_20261120_151227/summary.json) | 0.7663 ± 0.0837 | −0.55 | 70.75 | 69.48 | 62.50 | 76.47 |
| [HR Attention + Max](s123_hrk4_noema_hrattnmax_5fold_20261122_012712/summary.json) | 0.7489 ± 0.0748 | −2.29 | 70.99 | 69.95 | 64.38 | 75.52 |
| [Layer3 局部 LR–HR 对齐](s123_hrk4_noema_layer3aligned_5fold_20261121_121944/summary.json) | 0.7471 ± 0.0491 | −2.48 | 72.00 | 70.60 | 63.13 | 78.08 |
| [逐 Region 融合 Gate](s123_hrk4_noema_regionfusiongate_5fold_20261120_235722/summary.json) | 0.7115 ± 0.0763 | −6.03 | 65.16 | 64.27 | 59.38 | 69.16 |
| [K=4 HR 选块取消间距限制](s123_hrk4_noema_nospacing_5fold_20261123_213925/summary.json) | 0.7373 ± 0.0673 | −3.46 | 67.70 | 68.37 | 71.88 | 64.87 |
| [LR/HR 独立投影](s123_hrk4_noema_scaleproj_5fold_20261124_101720/summary.json) | **0.7907 ± 0.0657** | +1.89 | 71.78 | 69.44 | 56.88 | 82.01 |

LR/HR 独立投影的平均 Test AUC 最高，提升主要来自 Fold 4（0.6861 → 0.8400）。Layer3 对齐的 BACC 最高。此前四组模型消融的逐折结果、训练轨迹和复核范围见[第四周消融分析](analysis/k4_noema_ablation.md)。

## 实验记录

各组沿用原始运行名，组级汇总保存为 `summary.json`，逐折记录保存于 `fold_0` 至 `fold_4`。

- `config.yaml`、`data_provenance.json`、`run_info.json`
- `history.csv`、`best_val_predictions.csv`
- `test_metrics.json`、`test_predictions.csv`
- HR Attention + Max 的 `hr_attn_max_diagnostics.json`

以上共 221 份原始记录。模型权重与 checkpoint 保留在本地。
