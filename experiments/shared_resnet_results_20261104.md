# 2026-11-04 共享 ResNet 实验记录

本次同步新增的原始运行记录与逐轮日志，不包含任何模型权重。下列4组实验仍使用 attention + mean + topk 三路 MIL；新增 mean-only 配置属于下一组消融，本地尚未收到该配置的训练结果。

## 已有五折 SR0 整批结果

使用 seed=1、最多24个训练 region、BF16、300轮上限及原早停设置。这里只收到 Fold 0、1、2，不能作为完整五折结果；Fold 3、4 尚无本地记录。

| Fold | 状态 | 已完成轮数 | 所选 epoch | AUC | ACC | BACC | 肿瘤敏感度 | 正常特异度 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | early_stopped | 33 | 21 | 0.640762 | 0.660377 | 0.610704 | 0.318182 | 0.903226 |
| 1 | early_stopped | 40 | 28 | 0.752199 | 0.792453 | 0.756598 | 0.545455 | 0.967742 |
| 2 | early_stopped | 35 | 23 | 0.777126 | 0.754717 | 0.757331 | 0.772727 | 0.741935 |

Fold 0、1 日志含最终 early_stopped 行；Fold 2 的 history.csv 有35轮，run_info.json 记录 early_stopped，但所收到日志仅包含到第25轮的完整指标及第26轮的部分 WSI。保留原始日志和元数据，不补写缺失内容。

## SR 前8轮实验快照

shared_resnet18_512_sr01_8ep_f2_300_20261103_111814 使用三折划分的 Fold 2、region 微批次2、前8轮 SR=0.1，随后为0。history.csv 有38轮，日志进入第39轮但无结束行，run_info.json 仍为 incomplete，因此保留未完成状态。

按训练时 min_delta=0.001 的选轮规则，现有最佳验证预测对应第27轮（AUC 0.759671）。历史第28轮原始 AUC 为0.759936，但提升不足0.001，不替换所选 checkpoint。该结果不能与五折 Fold 2 直接视作同一划分的对照。

## 文件索引与解释范围

- [共享模型结果索引](../results/summary_shared_resnet.csv)包含配置、history、最佳验证预测及日志路径。AUC 已由最佳验证预测重算核对；其余指标取同一所选 epoch。
- 所选 epoch 与分类阈值使用验证集确定，相关指标不等同于独立测试性能；记录中的 Git commit 是训练元数据原值，不保证服务器当时没有未提交修改。
- 根目录 predictions.csv 按原文件上传，但缺少配套运行来源，不归入上述实验或结果汇总。
- 同步之前 SR0 三折 Fold 2 的两张训练曲线图，原图放在对应 runs 目录；其他历史记录不改写。

## 新代码与后续配置

- 共享 ResNet 的 classification.pooling 支持 attention、mean、topk 的任意非空组合，默认保持原三路；SR 实现保持原样。
- 训练、评估支持独立的3折/5折方案，附上 native512_264 的5份划分及生成脚本。
- [mean-only / SR0 / 整批配置](../configs/native512_shared_resnet18_mean_sr0_full.yaml)保留原优化与增强参数；其训练结果尚不在本次上传内容中。
