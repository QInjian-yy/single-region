# 旧划分补跑：MeanMIL＋GCSAM Fold 3–4

归档核验时间：2026-11-13T19:56:04.680575+08:00。本目录保存264张WSI旧内部五折的Train/Val补跑结果。Fold 4仍在服务器训练，用户要求先归档现有记录；**本地Fold 4是运行中快照，状态保持incomplete**。

| Fold | 状态 | 本地最后epoch | 所选最佳epoch | Val AUC | Train AUC（同epoch） | Train/Val WSI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3 | early_stopped | 65 | 53 | 0.823188 | 0.866731 | 211/53 |
| 4 | incomplete | 57 | 56 | 0.900758 | 0.852882 | 212/52 |

Fold 3已经正常早停；Fold 4的最佳epoch由本地history按原min_epochs=5、min_delta=0.001规则重放得到，并由best_val_predictions.csv复算验证。Fold 4原run_info.json尚无最终best字段，未改写原始文件。

## 与上周三折对应

| Fold | 原先已完成的Val AUC | 最佳epoch | 原始记录 |
| --- | ---: | ---: | --- |
| 0 | 0.678886 | 49 | [gcsam_f0_20261106_095824](../../runs/gcsam_f0_20261106_095824) |
| 1 | 0.857771 | 51 | [gcsam_f1_3090_repeat_20261106_155902](../../runs/gcsam_f1_3090_repeat_20261106_155902) |
| 2 | 0.725073 | 12 | [gcsam_fold2_3090](../../runs/gcsam_fold2_3090) |

Fold 3–4与上周原Fold 1配置逐项对照，仅class_weight_values随训练标签数量变化。原Fold 0的manifest路径首字母大小写不同，实际文件记录哈希一致。manifest、标签、预处理报告的记录SHA-256一致；各自split SHA-256与本地downstream_train/native512_264中的对应旧折文件一致。模型是MeanMIL＋ResNet18仅微调layer4，GCSAM＋AdamW，rho=0.05，SR=0，EMA=0.999，seed=1，全量Region整批编码。保留原balanced损失配置。

本组是旧协议验证结果，没有Test集合。CAMIL的8组40折另见[第二周主页](../README.md)。待Fold 4完成并下载最新记录后，再更新本快照及旧五折最终汇总。

## 文件

每折只归档config.yaml、history.csv、run_info.json、data_provenance.json、best_val_predictions.csv。所有原始文件按字节复制，排除best.pth、last.pth及其他权重。新增核验数据见[核验记录](../audit_additions_20261113.json)。
