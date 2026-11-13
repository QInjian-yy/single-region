# CAMIL 划分：7组完整五折实验对比

核对日期：2026-11-13。本目录独立归档第二周的 7 组、35 折记录。

当前结果：ASRD+AEM 的五折平均 Test AUC 最高（0.669788），与 ASRD（0.668913）仅差 0.000874。新增 SR 前10轮和分类器 Dropout=0.3 均降低各自对照的平均 Test AUC。SR 组在所选验证阈值下的 Test BACC 和肿瘤召回率最高，体现指标之间的取舍。

## 统计口径与核验

- 每折读取最终用于 Test 的 best checkpoint 对应 epoch；Train/Val 来自该 epoch 的 history.csv，Test 来自 test_predictions.csv 独立重算。Train 指训练集在评估模式下的指标。
- 表中均值为五折等权宏平均；± 为五折样本标准差（ddof=1），不是置信区间。相同训练 seed=1，五折来自固定划分，不能当作五次独立随机种子重复。
- 70份 Val/Test 逐片预测的 AUC 经 sklearn 重算，最大差值为0；ACC、BACC、敏感度、特异度及预测类别也逐项核对一致。
- 35份 run_info 均为 early_stopped，epoch连续、数值有限；按 min_epochs=5、patience=12、min_delta=0.001 重放选择/早停，35折均与记录一致。
- 各组对应折的划分、标签、source index、两个源 manifest 的记录哈希一致；本地 protocol、source index、labels 和五折 CSV 的文件哈希与记录一致。
- 按五折 CSV 与逐片预测核对：每折 Train/Val/Test 的 WSI ID 无重叠，测试集覆盖393张不同WSI，每张恰好测试一次。该检查只验证 WSI 级分离，未验证患者级独立性。
- 每折阈值由其 Val 预测选择，并原样用于 Test；35折全部匹配。Train ACC/BACC 使用0.5阈值，不能把其与自适应阈值的 Val/Test ACC 差值直接等同泛化差距。
- 分析未重跑图像推理，也未验证服务器当时所有未提交源码的逐字一致性。Train AUC来自历史记录；Val/Test AUC来自独立数值复算。

## 三个集合的 AUC

| 实验 | Train AUC | Val AUC | Test AUC |
| --- | --- | --- | --- |
| Attention + GCSAM | 0.814825 ± 0.078092 | 0.667773 ± 0.084068 | 0.624621 ± 0.077905 |
| Attention + AdamW | 0.894669 ± 0.097034 | 0.680968 ± 0.067409 | 0.663155 ± 0.080060 |
| Mean + AdamW | 0.849320 ± 0.097817 | 0.705072 ± 0.052054 | 0.662001 ± 0.072291 |
| Attention + AdamW + ASRD | 0.858935 ± 0.090371 | 0.672895 ± 0.068028 | 0.668913 ± 0.080693 |
| Attention + AdamW + ASRD + AEM | 0.863210 ± 0.092527 | 0.691547 ± 0.075034 | 0.669788 ± 0.081876 |
| Attention + AdamW + ASRD + SR前10轮 | 0.821061 ± 0.094688 | 0.673782 ± 0.046760 | 0.655371 ± 0.083730 |
| Attention + AdamW + 分类器Dropout 0.3 | 0.861709 ± 0.079141 | 0.661991 ± 0.062626 | 0.649232 ± 0.096856 |

## 逐折 Test AUC

| 实验 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | --- | --- | --- | --- | --- |
| Attention + GCSAM | 0.577460 | 0.666556 | 0.577460 | 0.742188 | 0.559443 |
| Attention + AdamW | 0.705785 | 0.705120 | 0.641622 | 0.730639 | 0.532609 |
| Mean + AdamW | 0.651596 | 0.697473 | 0.640625 | 0.757812 | 0.562500 |
| Attention + AdamW + ASRD | 0.700133 | 0.736702 | 0.650598 | 0.720448 | 0.536685 |
| Attention + AdamW + ASRD + AEM | 0.654920 | 0.736037 | 0.635971 | 0.763247 | 0.558764 |
| Attention + AdamW + ASRD + SR前10轮 | 0.655918 | 0.672872 | 0.638963 | 0.771739 | 0.537364 |
| Attention + AdamW + 分类器Dropout 0.3 | 0.657247 | 0.740691 | 0.606715 | 0.734035 | 0.507473 |

## Test 阈值指标（五折均值，%）

| 实验 | ACC | BACC | 敏感度/肿瘤召回 | 特异度 |
| --- | --- | --- | --- | --- |
| Attention + GCSAM | 58.04 | 56.94 | 50.62 | 63.26 |
| Attention + AdamW | 60.30 | 59.81 | 56.88 | 62.74 |
| Mean + AdamW | 59.30 | 58.51 | 53.75 | 63.27 |
| Attention + AdamW + ASRD | 60.31 | 60.69 | 62.50 | 58.87 |
| Attention + AdamW + ASRD + AEM | 63.36 | 61.34 | 50.00 | 72.67 |
| Attention + AdamW + ASRD + SR前10轮 | 61.59 | 62.65 | 68.12 | 57.18 |
| Attention + AdamW + 分类器Dropout 0.3 | 61.33 | 59.11 | 46.88 | 71.35 |

## 配置对照与消融

共同配置：Native512，ImageNet ResNet18，仅训练layer4，冻结BN，256维Embedding，全量Region整批输入；bf16、seed=1、EMA=0.999、head_lr=0.0002、encoder_lr=0.00001、weight_decay=0.01、label_smoothing=0.1。VIB关闭；翻转、90度旋转及颜色扰动相同。class_weight 字段均为 balanced；每折按训练标签计算的 class_weight_values 在各组之间一致。每组跨折配置仅该权重数组变化。

对照比较逐折核对了全部 YAML 字段，差异如下：

- Attention + GCSAM → Attention + AdamW：平均 Test AUC 差 +0.038533；3/5 折提升。
  - `gcsam.base_optimizer`：`"adamw"` → `"<absent>"`。
  - `gcsam.enabled`：`true` → `"<absent>"`。
  - `gcsam.rho`：`0.05` → `"<absent>"`。
  - `gcsam.start_epoch`：`1` → `"<absent>"`。
  - `optimizer`：`"gcsam"` → `"adamw"`。
- Attention + AdamW → Mean + AdamW：平均 Test AUC 差 -0.001153；2/5 折提升。
  - `classification.pooling`：`["attention"]` → `["mean"]`。
- Attention + AdamW → Attention + AdamW + ASRD：平均 Test AUC 差 +0.005759；3/5 折提升。
  - `classification.region_interaction.enabled`：`"<absent>"` → `true`。
  - `classification.region_interaction.gamma_init`：`"<absent>"` → `0.0`。
  - `classification.region_interaction.rank`：`"<absent>"` → `16`。
  - `classification.region_interaction.temperature`：`"<absent>"` → `1.0`。
  - `classification.region_interaction.topk`：`"<absent>"` → `3`。
  - `classification.region_interaction.type`：`"<absent>"` → `"asrd"`。
- Attention + AdamW + ASRD → Attention + AdamW + ASRD + AEM：平均 Test AUC 差 +0.000874；2/5 折提升。
  - `classification.aem.anneal_epochs`：`"<absent>"` → `50`。
  - `classification.aem.enabled`：`"<absent>"` → `true`。
  - `classification.aem.weight`：`"<absent>"` → `0.01`。
- Attention + AdamW + ASRD → Attention + AdamW + ASRD + SR前10轮：平均 Test AUC 差 -0.013542；2/5 折提升。
  - `lambda_sr`：`0.0` → `0.05`。
  - `training.sr_stop_after_epochs`：`3` → `10`。
- Attention + AdamW → Attention + AdamW + 分类器Dropout 0.3：平均 Test AUC 差 -0.013922；2/5 折提升。
  - `classification.classifier_dropout`：`0.0` → `0.3`。

配置对照可以隔离所比较的训练策略；SR裁剪、Dropout等会改变随机数消耗，相同seed不代表逐步使用完全相同的随机样本序列。

## 两组新增实验的解释

**SR前10轮组**：正确对照为 Attention+AdamW+ASRD；没有AEM。全部五折的history确认：epoch1–10为lambda_sr=0.05且SR loss大于0，epoch11起lambda_sr和SR loss均为0。总损失与分类损失加加权SR损失的最大数值差约4.85e-9。Test AUC下降0.013542，Fold0/1/2下降，Fold3提高0.051291，Fold4仅提高0.000679。

该组 best epoch 为 [65,5,5,5,10]，停止 epoch 为 [77,17,17,17,22]。五折均跑过SR关闭阶段，但其中四折的最终测试模型仍选自SR开启阶段。因此不能表述为“五折都使用完成SR后继续分类微调的模型评估”。

在各自Val阈值下，SR使Test敏感度从62.50%升至68.125%，特异度从58.87%降至57.18%。393张测试WSI累计混淆计数：ASRD TP100/FN60/TN137/FP96；SR TP109/FN51/TN133/FP100。BACC从60.69%升至62.65%，与AUC下降并不矛盾。

**分类器Dropout=0.3组**：正确对照是普通 Attention+AdamW；未启用ASRD/AEM/SR，仅 classifier_dropout 从0改0.3。Test AUC从0.663155降至0.649232（-0.013922），Val从0.680968降至0.661991。Train AUC下降，但验证和测试均未获益，当前证据不能支持这项正则改善泛化。Fold1/3提高，Fold0/2/4下降，Fold4为0.507473。

**AEM**：相对ASRD，5折所选Val AUC均提高，但Test仅2/5提高，平均增益仅0.000874。ACC提高伴随肿瘤召回率由62.50%降至50%，不能仅凭ACC称其全面更好。

**ASRD与基础模型**：相对Attention+AdamW，ASRD的平均Test AUC提高0.005759（3/5折提高），是小幅观察性提升，尚无多seed复现证据。Mean+AdamW的Val平均最高0.705072，Test与Attention+AdamW接近；当前GCSAM配置的Test均值低于AdamW约0.038533。

## 选择时点、过拟合和共同难点

| 实验 | best epoch（Fold0–4） | stop epoch（Fold0–4） | Train–Val AUC均值差 |
| --- | --- | --- | --- |
| Attention + GCSAM | 29, 58, 6, 5, 44 | 41, 70, 18, 17, 56 | 0.147053 |
| Attention + AdamW | 62, 56, 5, 12, 62 | 74, 68, 17, 24, 74 | 0.213700 |
| Mean + AdamW | 31, 12, 5, 7, 37 | 43, 24, 17, 19, 49 | 0.144248 |
| Attention + AdamW + ASRD | 30, 17, 5, 7, 37 | 42, 29, 17, 19, 49 | 0.186039 |
| Attention + AdamW + ASRD + AEM | 68, 18, 5, 5, 16 | 80, 30, 17, 17, 28 | 0.171662 |
| Attention + AdamW + ASRD + SR前10轮 | 65, 5, 5, 5, 10 | 77, 17, 17, 17, 22 | 0.147280 |
| Attention + AdamW + 分类器Dropout 0.3 | 60, 33, 5, 23, 9 | 72, 45, 17, 35, 21 | 0.199718 |

训练AUC比验证AUC高的现象仍广泛存在。不同组的best时点不同，不能只靠Train–Val均值差缩小就断言正则有效。SR组Fold4由epoch10的Train/Val=0.8958/0.6350走到epoch22的0.9638/0.5894，属于训练继续改善而验证恶化的具体例子。

当前 min_epochs=5 同时约束最佳模型保存：第1–4轮即使Val AUC更高也不会成为best。已有权重中最高Val为AEM组Fold2、epoch5、0.776155；全部history的原始最高Val为ASRD组Fold2、epoch2、0.799592，后者未被当前保存规则选中。以上是规则的实际行为，不能据此推断未保存模型的Test结果。

Fold4在7组中的Test AUC均最低，范围0.5075–0.5625；Fold2常出现Val较高而Test一般，Fold3则常出现相反现象。差异提示结果受具体折影响明显，当前结果不支持仅增加正则即可解决这些问题。

## 记录耗时与边界

| 实验 | 五折累计记录时长（小时） |
| --- | --- |
| Attention + GCSAM | 7.38 |
| Attention + AdamW | 5.99 |
| Mean + AdamW | 3.53 |
| Attention + AdamW + ASRD | 3.63 |
| Attention + AdamW + ASRD + AEM | 4.00 |
| Attention + AdamW + ASRD + SR前10轮 | 25.88 |
| Attention + AdamW + 分类器Dropout 0.3 | 4.40 |

时长是run_info起止时间之和，受早停轮数、并发任务和I/O影响，不能当作受控单步速度对比。SR组累计约25.88小时，ASRD对照约3.63小时。

## 源目录

- Attention + GCSAM：[源记录](camil_attn_3090_20261110_104709)。
- Attention + AdamW：[源记录](camil_attn_adamw_3090_20261110_193735)。
- Mean + AdamW：[源记录](camil_mean_adamw_5fold_20261111)。
- Attention + AdamW + ASRD：[源记录](camil_attn_adamw_asrd_3090_20261111_102911)。
- Attention + AdamW + ASRD + AEM：[源记录](camil_attn_adamw_asrd_aem_20261111_154122)。
- Attention + AdamW + ASRD + SR前10轮：[源记录](camil_attn_adamw_asrd_sr0p05_first10_20261111_204023)。
- Attention + AdamW + 分类器Dropout 0.3：[源记录](camil_attn_adamw_clsdrop03_5fold)。

输出附带 summary.csv（七组汇总）、fold_metrics.csv（35折详细结果，包括Train/Val/Test、阈值、混淆计数、best/stop epoch）、archive_manifest.json（归档文件SHA-256）。归档中的原始实验记录保持字节一致；未包含模型权重。
