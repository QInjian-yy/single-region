# 第四周实验

新整理并发布的实验记录归档到本目录。此前的结果见[第三周实验](../week3_experiments/README.md)。

## 测试集指标

Test AUC 为五折等权平均 ± 样本标准差。ACC、BACC、肿瘤召回率和特异度为五折等权平均，单位为 %。各折使用在验证阶段保存的判定阈值。

| 实验与记录目录 | Test AUC | ACC | BACC | 肿瘤召回率 | 特异度 |
| --- | ---: | ---: | ---: | ---: | ---: |

## 实验记录

每组沿用原始运行名，组级汇总保存为 `summary.json`。逐折记录放在单层的 `fold_0` 至 `fold_4` 目录中，保留以下原始文件。

- `config.yaml`、`data_provenance.json`、`run_info.json`
- `history.csv`、`best_val_predictions.csv`
- `test_metrics.json`、`test_predictions.csv`

实验对照分析可放在 `analysis/`。阶段性记录注明实际完成状态和折数。模型权重和 checkpoint 保留在本地。
