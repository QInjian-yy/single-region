# 第四周实验

本周完成 11 组 CAMELYON16 Seed123 五折消融，共 55 折。分尺度 BN、HR Layer3 残差、投影后学习率减半和融合前分路 LayerNorm 分别在 LR/HR 独立投影上改动一处，其余组以原始 K=4 NoEMA 为对照。此前的结果见[第三周实验](../week3_experiments/README.md)。

## 测试集指标

Test AUC 为五折等权平均 ± 样本标准差。表中的 Δ AUC 为相对原始 K=4 NoEMA 基线的百分点变化。ACC、BACC、肿瘤召回率和特异度为五折等权平均（%），各折沿用验证集选出的判定阈值。

| 实验与汇总记录 | Test AUC | Δ AUC（百分点） | ACC | BACC | 肿瘤召回率 | 特异度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [K=4 NoEMA 基线](../week3_experiments/s123_hrk4_noema_5fold_20261119_092239/summary.json) | 0.7719 ± 0.0661 | — | 67.95 | 66.55 | 58.75 | 74.36 |
| [Region 内 HR 自注意力](s123_hrk4_noema_hrselfattn_h4_gate0_5fold_20261120_151227/summary.json) | 0.7663 ± 0.0837 | −0.55 | 70.75 | 69.48 | 62.50 | 76.47 |
| [HR Attention + Max](s123_hrk4_noema_hrattnmax_5fold_20261122_012712/summary.json) | 0.7489 ± 0.0748 | −2.29 | 70.99 | 69.95 | 64.38 | 75.52 |
| [Layer3 局部 LR–HR 对齐](s123_hrk4_noema_layer3aligned_5fold_20261121_121944/summary.json) | 0.7471 ± 0.0491 | −2.48 | 72.00 | 70.60 | 63.13 | 78.08 |
| [逐 Region 融合 Gate](s123_hrk4_noema_regionfusiongate_5fold_20261120_235722/summary.json) | 0.7115 ± 0.0763 | −6.03 | 65.16 | 64.27 | 59.38 | 69.16 |
| [K=4 HR 选块取消间距限制](s123_hrk4_noema_nospacing_5fold_20261123_213925/summary.json) | 0.7373 ± 0.0673 | −3.46 | 67.70 | 68.37 | 71.88 | 64.87 |
| [LR/HR 独立投影](s123_hrk4_noema_scaleproj_5fold_20261124_101720/summary.json) | **0.7907 ± 0.0657** | +1.89 | 71.78 | 69.44 | 56.88 | 82.01 |
| [Spatial Top-4](s123_hrk4_noema_spatialtopk4_5fold_20261124_105733/summary.json) | 0.7422 ± 0.1028 | −2.97 | 70.74 | 69.07 | 60.00 | 78.13 |
| [独立投影 + Layer4 分尺度 BN](s123_hrk4_noema_scaleproj_layer4splitbn_5fold_20261124_164152/summary.json) | 0.7813 ± 0.0790 | +0.94 | 71.51 | 69.67 | 60.00 | 79.34 |
| [独立投影 + HR Layer3 残差](s123_hrk4_noema_scaleproj_hrlayer3_5fold_20261124_213308/summary.json) | 0.7460 ± 0.1118 | −2.58 | 65.18 | 65.26 | 65.62 | 64.89 |
| [独立投影 + 投影后学习率减半](s123_hrk4_noema_scaleproj_postprojhalflr_5fold_20261125_100204/summary.json) | 0.7750 ± 0.0680 | +0.32 | 70.02 | 70.03 | 70.00 | 70.06 |
| [独立投影 + 融合前分路 LayerNorm](s123_hrk4_noema_scaleproj_splitfusionln_5fold_20261125_100204/summary.json) | 0.7730 ± 0.0645 | +0.11 | 74.55 | 70.97 | 51.88 | 90.06 |

LR/HR 独立投影的平均 Test AUC 最高，提升主要来自 Fold 4（0.6861 → 0.8400）。融合前分路 LayerNorm 的 BACC 最高。此前四组模型消融的逐折结果、训练轨迹和复核范围见[第四周消融分析](analysis/k4_noema_ablation.md)。

Spatial Top-4 的 AUC 下降主要来自 Fold 3（0.7620 → 0.6307）和 Fold 1（0.7357 → 0.6795）。按各自验证集选定的阈值，五折平均 BACC 从 66.55% 升至 69.07%。

与直接对照的 LR/HR 独立投影相比，分尺度 BN 和 HR Layer3 残差的平均 Test AUC 分别低 0.94 和 4.47 个百分点。两组的主要降幅均出现在 Fold 1 和 Fold 3。按各自验证集选定的阈值，分尺度 BN 的肿瘤召回率为 60.00%，HR Layer3 残差为 65.625%，对应特异度为 79.34% 和 64.89%。

投影后学习率减半组在 LR/HR 独立投影的基础上，将投影后的学习率从 2e-4 调为 1e-4。五折平均 Test AUC 为 0.7750，较直接对照低 1.57 个百分点，降幅主要出现在 Fold 1 和 Fold 2。两组的五折数据来源及 HR 像素缓存记录一致。

融合前分路 LayerNorm 组在融合前分别对 LR 特征和聚合后的 HR 特征做 LayerNorm。与 LR/HR 独立投影的配置、逐折划分和 HR 像素缓存记录一致。五折平均 Test AUC 为 0.7730，较直接对照低 1.77 个百分点，主要降幅出现在 Fold 0、1 和 4。按各折验证集选定的阈值，BACC 由 69.44% 升至 70.97%，特异度由 82.01% 升至 90.06%，肿瘤召回率由 56.88% 降至 51.88%。

## 实验记录

各组沿用原始运行名，组级汇总保存为 `summary.json`，逐折记录保存于 `fold_0` 至 `fold_4`。

- `config.yaml`、`data_provenance.json`、`run_info.json`
- `history.csv`、`best_val_predictions.csv`
- `test_metrics.json`、`test_predictions.csv`
- HR Attention + Max 的 `hr_attn_max_diagnostics.json`

以上共 401 份原始记录。模型权重与 checkpoint 保留在本地。
