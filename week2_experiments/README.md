# 第二周实验：CAMIL 划分五折对比

本目录独立保存第二周的 **8组CAMIL完整五折实验、40折，以及1组旧划分Fold 3–4补跑记录**，供周报汇报与结果核对使用。归档日期：2026-11-13。

[完整对比报告](comparison.md) · [8组CAMIL汇总 CSV](summary.csv) · [40折CAMIL详细指标 CSV](fold_metrics.csv)

CAMIL各组包含 Fold 0–4 的原始配置、训练历史、验证预测、测试预测、测试指标、运行信息、数据来源记录，以及五折 summary.json。**不含 best.pth、last.pth 或任何模型权重。**

| 实验与记录目录 | Train AUC | Val AUC | Test AUC（均值 ± 样本标准差） |
| --- | ---: | ---: | ---: |
| [Attention + GCSAM](camil_attn_3090_20261110_104709) | 0.8148 | 0.6678 | 0.6246 ± 0.0779 |
| [Attention + AdamW](camil_attn_adamw_3090_20261110_193735) | 0.8947 | 0.6810 | 0.6632 ± 0.0801 |
| [Mean + AdamW](camil_mean_adamw_5fold_20261111) | 0.8493 | 0.7051 | 0.6620 ± 0.0723 |
| [Attention + AdamW + ASRD](camil_attn_adamw_asrd_3090_20261111_102911) | 0.8589 | 0.6729 | 0.6689 ± 0.0807 |
| [Attention + AdamW + ASRD + AEM](camil_attn_adamw_asrd_aem_20261111_154122) | 0.8632 | 0.6915 | 0.6698 ± 0.0819 |
| [Attention + AdamW + ASRD + SR前10轮](camil_attn_adamw_asrd_sr0p05_first10_20261111_204023) | 0.8211 | 0.6738 | 0.6554 ± 0.0837 |
| [Attention + AdamW + 分类器Dropout 0.3](camil_attn_adamw_clsdrop03_5fold) | 0.8617 | 0.6620 | 0.6492 ± 0.0969 |
| [Attention + AdamW + 冻结ResNet18](camil_attn_adamw_frozen_5fold) | 0.8390 | 0.6480 | 0.6605 ± 0.0840 |

另附：[旧划分Fold 3–4补跑快照](gcsam_mean_legacy5fold_f3_f4_20261113_145057/README.md)。Fold 3已早停；Fold 4仍在训练，本地快照到epoch 57，保持`incomplete`。该目录使用264张WSI旧Train/Val协议，未纳入上述CAMIL的Test汇总。

Train/Val取每折最终用于测试的最佳权重所在epoch。各折仅按Val选择模型和分类阈值，再评估Test。均值按五折等权计算；这五折使用同一个训练seed=1，并非五个训练随机种子的重复实验。

## 周报可用结论

- ASRD相对Attention+AdamW的平均Test AUC提高0.0058；3/5折提高，属于本次实验的小幅正向结果。
- ASRD+AEM的Test AUC为0.6698，与ASRD的0.6689基本持平；Val改善没有转换为稳定的Test增益。
- SR前10轮组相对ASRD降低平均Test AUC约0.0135，但Test BACC和肿瘤召回率提高，体现指标取舍。
- 分类器Dropout=0.3相对普通Attention+AdamW降低平均Test AUC约0.0139；本次没有体现泛化改善。
- Mean+AdamW平均Val AUC最高0.7051；所有配置的Fold4 Test均表现较弱。
- 冻结ResNet18组与Attention+AdamW只差`shared_encoder.trainable: layer4 → frozen`；Test AUC为0.6605，对照为0.6632，本次差异很小。

## 消融对照

- Attention+GCSAM ↔ Attention+AdamW：优化器策略。
- Mean+AdamW ↔ Attention+AdamW：MIL聚合方式。
- Attention+AdamW ↔ 冻结ResNet18：仅训练layer4与冻结整个编码器的对照。
- Attention+AdamW ↔ Attention+AdamW+ASRD：Region间关系交互。
- Attention+AdamW+ASRD ↔ 加AEM：注意力熵正则，初始权重0.01、50轮退火。
- Attention+AdamW+ASRD ↔ 加SR前10轮：epoch1–10的lambda_sr=0.05，之后为0；不含AEM。
- Attention+AdamW ↔ 分类器Dropout 0.3：仅classifier_dropout从0变为0.3；不含ASRD/AEM/SR。

## 划分与核验

- 使用现有[固定CAMIL式划分](../data_protocols/camil_5fold)：393张WSI、五折轮换Test、每折约60%训练/20%验证/20%测试。
- 8组CAMIL实验的对应划分和数据来源记录一致。每张WSI恰好进入一次Test；协议只验证WSI级分离，未核实患者级独立性。
- 80份CAMIL Val/Test逐片预测经sklearn复算，AUC与记录一致；阈值指标也一致。
- 40折CAMIL实验均正常早停；按min_epochs=5、patience=12、min_delta=0.001重放，与记录一致。
- SR五折best epoch为65、5、5、5、10：其中4折的最佳模型来自SR仍开启的阶段，不能称为全部使用关闭SR后继续微调的模型。

## 文件组织

```text
week2_experiments/
  README.md
  comparison.md
  summary.csv
  fold_metrics.csv
  archive_manifest.json
  <CAMIL实验目录>/
    summary.json
    fold_0/ ... fold_4/
      config.yaml
      history.csv
      best_val_predictions.csv
      test_metrics.json
      test_predictions.csv
      run_info.json
      data_provenance.json
  gcsam_mean_legacy5fold_f3_f4_20261113_145057/
    README.md
    fold_3/
    fold_4/  # incomplete，运行中快照
  audit_additions_20261113.json
```

原先存放在仓库runs/中的本批4组记录迁入本目录，另3组在此补充归档。旧提交历史仍可追溯，本地原始runs/与权重保留。这里是独立的实验结果归档，未复制服务器训练源码或数据图像。文件SHA-256见[归档清单](archive_manifest.json)。

本次补充归档46份原始非权重记录：冻结组36份，旧Fold 3–4共10份。后者仅保留本次下载快照；Fold 4完成后需以最新记录更新。新增数据核验见[audit_additions_20261113.json](audit_additions_20261113.json)。
