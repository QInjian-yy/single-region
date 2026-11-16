# 第三周实验：Attention、AEM 与多尺度输入

本目录独立归档第三周的 **7 组完整五折实验、35 折**，与[第二周实验](../week2_experiments/README.md)对应。归档日期：2026-11-16。

每组保存 36 份原始实验记录，共 252 份：五折汇总 `summary.json`，以及 Fold 0–4 的配置、训练历史、运行信息、数据来源、最佳验证预测、测试指标和测试预测。原始记录保持原样。**模型权重仅保留在本地，不上传 best.pth、last.pth、*.pt、*.ckpt 或 *.safetensors。**

| 实验与记录目录 | Train AUC | Val AUC | Test AUC（均值 ± 样本标准差） |
| --- | ---: | ---: | ---: |
| [单尺度 Attention + AEM](s123_attn_aem_20261114_172804) | 0.9093 | 0.6987 | 0.7025 ± 0.0739 |
| [单尺度 Attention + AEM + 分类器 Dropout 0.3](s123_aem_drop03_20261115_013644) | 0.8528 | 0.6888 | 0.7018 ± 0.0668 |
| [多尺度 + AEM 0.01 + EMA 0.999，SR=0](s123_multiscale_only_sr0_20261115_120923) | 未记录 | 0.7345 | **0.7345 ± 0.0659** |
| [多尺度 + AEM + 全程 SR 0.05](s123_multiscale_aem_sr005_20261115_181153) | 未记录 | 0.7313 | 0.7157 ± 0.1055 |
| [多尺度 + AEM，关闭 EMA](s123_multiscale_aem_noema_sr0_20261115_194946) | 未记录 | 0.7441 | 0.7052 ± 0.0538 |
| [多尺度 + AEM 降至 0.005](s123_multiscale_aem005_sr0_20261116_004006) | 未记录 | 0.7177 | 0.7146 ± 0.1012 |
| [多尺度 + AEM + EMA 改为 0.99](s123_multiscale_ema099_sr0_20261116_004012) | 未记录 | 0.7392 | 0.7238 ± 0.0521 |

Train/Val 取每折最终测试所用最佳权重所在 epoch，按五折等权平均。多尺度组未记录 Train AUC，不能据此判断训练/验证差距是否缩小。Test 的标准差按五折样本标准差计算；训练 seed 均为 1，并非五个随机种子的重复实验。

## 实验配置与对照

- 七组均配置 AttentionMIL + AdamW、ResNet18 ImageNet 预训练、仅微调 layer4、冻结 BN、BF16，并启用 AEM；未启用 ASRD 或 GCSAM。
- 除明确列出的变化外，AEM 初始权重为 0.01、50 轮余弦退火，EMA 为 0.999，SR 权重为 0。
- 单尺度 AEM 与 Dropout 组的配置仅差 `classification.classifier_dropout: 0 → 0.3`。
- 四个多尺度消融组分别与多尺度 SR0 对照比较：全程 `lambda_sr: 0.05`、`ema_decay: 0`、AEM 权重 `0.005`、`ema_decay: 0.99`；对应 YAML 各仅改变这一项。
- `multiscale_only_sr0` 名字中的 `only` 不代表关闭 AEM；该组实际启用了 AEM 0.01。
- 多尺度记录与权重元数据表明：LR/HR 共享 ResNet18，每个 Region 使用 2 个 256×256 HR 小块，推理时使用 HR；缓存 HR 像素，不缓存特征。因此，SR=0 的多尺度组仍使用真实 HR 信息。
- 多尺度 SR0 与单尺度 AEM 还存在训练入口及 Region microbatch 配置差别；不能把其全部差异解释成某一个融合参数的独立贡献。

## 划分与结果核验

- 本周来源记录指向服务器 `data_protocols/camil_5fold_seed123_sourcestrat`。七组对应的数据来源、标签和五份划分哈希一致；本周五份划分与第二周均不同，跨周分数差额不能全部归因于模型。
- 每组五折 Test 预测合计覆盖 393 张不同 WSI，每张恰好测试一次；同折 Val/Test 无重叠，七组对应折的预测名单一致。记录只声明 WSI 级分离，未核实患者级独立性；新划分原始 CSV 尚未同步本地，因此本次未完成 Train 名单核验。
- 70 份 Val/Test 逐片预测经 sklearn 重算，AUC、ACC、BACC、敏感性和特异性与记录一致；验证阈值选择与五折汇总也一致。
- 35 折均可按 `min_epochs=5`、`patience=12`、`min_delta=0.001` 重放出相同最佳 epoch 和结束 epoch，实际结束于 17–45 轮。当前规则不保存前四轮最佳模型，因此部分折选中的 Val AUC 低于全程原始最高值。
- 多尺度 SR0 相对同划分单尺度 AEM，4/5 折 Test AUC 提高，均值提高 0.0320；其平均 Test BACC 为 0.6605、肿瘤召回率为 50.63%，AUC 提升尚未转化为更好的阈值分类表现。

## 代码与复现范围

本次归档包含实验记录，不包含服务器新增实现。单尺度入口记录为 `train_e2e_camil_aem.py`；多尺度入口记录为 `multiscale_train.py`，checkpoint 格式为 `c16-e2e-lrhr-shared-resnet-sr-aem-v1`。

归档时本地尚缺多尺度训练入口及其模型、数据加载、HR 选块实现，也缺 `camil_5fold_seed123_sourcestrat` 完整划分目录。现有结果已做数值核验，多尺度融合公式、梯度路径及完整训练复现仍需服务器源码核查。
