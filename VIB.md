# 共享 ResNet 的可选 VIB

VIB 默认关闭。旧 YAML、MIL 选择、SR 开关和训练/评估入口继续使用；
关闭时不创建 VIB 参数、不采样，也不改变原损失或旧 checkpoint 的严格加载。

在配置副本的 `classification` 下增加，保留该项原有的其他字段：

```yaml
classification:
  vib:
    enabled: true
    beta: 1.0e-4
```

`beta` 是 KL 权重，必须有限且非负。`enabled: false` 或省略 `vib` 才恢复原路径；
开启 VIB 后仅设置 `beta: 0` 仍会执行映射和采样。

## 计算与维度

```text
同一 WSI 的 N 个 region
  → 原共享 ResNet → 原分类投影 h [N,D]
  → 两个 Linear(D,D)：mu、logvar [N,D]
  → 训练：z = mu + exp(0.5*logvar) * eps，eps ~ N(0,I)
     验证/推理：z = mu
  → 原选定的 MIL 分支 → 原分类器 → logits [1,2]
```

`D` 沿用 `classification.embedding_dim`，默认256；VIB不改变该维度。
N 是当前 WSI 实际选取的 region 数量，原 `max_train_regions` 规则继续生效。
整批模式仍只编码一次；microbatch 模式先拼成完整 `[N,D]`，再执行一次 VIB。
VIB 位于原 MIL 的 region dropout 之前，KL 对全部 N 个输入 region 计算。

KL 比较的是每个 region 的 `q(z|h)=N(mu,diag(exp(logvar)))` 和固定先验
`p(z)=N(0,I)`，不是比较 h 和 z 两个张量。计算方式为：

```text
KL = mean_regions(sum_features(0.5 * (mu² + exp(logvar) - 1 - logvar)))
total_loss = 原分类损失 + lambda_sr * 原SR损失 + beta * KL
```

指数、采样和 KL 归约使用 FP32；`expm1(logvar)` 等价实现 `exp(logvar)-1`，
改善零附近精度。保留原类权重、label smoothing、Dropout、优化器分组和 EMA。
VIB 参数进入原 head 优化组，并随 EMA 和 checkpoint 一起保存。
初始化 VIB 不消耗原模块的随机序列，因此相同 seed 下原模块的初值保持一致。

SR 解码器继续接收原四层共享特征，SR 到期关闭后 VIB 仍可训练。
开启 VIB 的梯度会更新可训练的共享编码器，因此不能保证训练后 SR 数值不变。
验证仍记录分类 CE 作为 `val_loss`，不混入 KL，保留原调度和选最优模型的指标含义。

## 使用与记录

示例 `configs/native512_shared_resnet18_mean_sr0_full_vib.yaml` 与现有
`native512_shared_resnet18_mean_sr0_full.yaml` 仅相差 VIB：MeanMIL、SR0、整批、
5fold、最多300轮等其他设置一致。`beta=1e-4` 是起始实验值，效果需要同条件对照。

沿用 `train_e2e_shared.py` 的原命令，仅替换配置和新输出目录；原 fold、data-root、
labels/split 参数按你的实验保留。现有任意合法 MIL 组合和 SR 开关均可与 VIB 配合。
评估沿用 `eval_e2e.py` 或 `eval_native512_reference.py`，SR 推理沿用 `infer_sr.py`。

开启时，WSI 日志和 `history.csv` 增加 `train_loss_kl`（未加权）、
`train_loss_vib`（beta*KL）；history 另记 `vib_beta`。其他原字段含义不变。
checkpoint 的 config 和 architecture 均记录 VIB，恢复时检查一致性并严格加载。
旧 checkpoint 缺少 VIB 字段时按关闭恢复，不将旧分类头自动迁移成 VIB。

新增检查命令：`python -m unittest tests.test_vib -v`。
覆盖 KL 分布计算、整袋归约、7种 MIL 组合、SR开关、整批/不等长microbatch、梯度、
优化器与 EMA、旧路径、确定性验证、训练入口日志与严格保存加载，以及可用时的 CUDA BF16。
这些检查不代表已证明抗过拟合或分类指标提升；正式结论需同划分、同 seed 的训练对照。

本机验证（2026-11-04）：全套139项测试通过（含新增8项）。与修改前 Git 版本
`1a0b81d` 对照，关闭 VIB 的整批/microbatch、SR开/关四种情况中，初始化、
随机数状态、训练损失、梯度和一次 AdamW 更新后的参数均逐值一致。
RTX 3060 上另用合成 `[2,3,512,512]`、D=256、scale=16、BF16 执行了
SR0/SR0.05 两次更新，KL为FP32且梯度有限，验证输出确定；未运行正式数据训练。
