# 独立 GCSAM + AdamW 实验

入口是 `train_e2e_gcsam.py`，两阶段训练在 `engine_gcsam.py`，优化器在
`optimizers/gcsam.py`。原 engine、训练入口、模型、Dataset、增强、划分、评估及 YAML
均不修改。原 AdamW/VIB 入口不导入或调用 GCSAM。

本次开发前的基线提交为 `82d4eb88814e8ba1929ffeabc3874564d1d04c00`。
开发前记录了412个已跟踪文件的 SHA-256，验收时逐文件复核。

## 论文与实现对应

- 论文：[arXiv:2501.11584v2](https://arxiv.org/html/2501.11584v2)，正文2.2、2.3节和公式(4)、(7)。
- IEEE Access最终版：[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12599882/)，III节和IV-D的ascent-only GC消融。
- 官方源码固定至提交
  [`29c1d96faa066e401e610d6a08c090bc345d27c6`](https://github.com/mhassann22/GCSAM/blob/29c1d96faa066e401e610d6a08c090bc345d27c6/cifar10_gcsam_resnet50.py)。

对一次WSI更新，令原参数为 w、原分类损失为 L：

1. 对已经选好Region并绑定增强参数的同一个sample，计算第一次 L(w) 和梯度 g。
2. 对多维梯度，沿 `tuple(range(1, grad.ndim))` 求均值并减去；一维bias、
   LayerNorm/BN affine梯度保持原值。未使用/冻结参数不参与扰动。
3. 将全部有效梯度组成一个全局L2范数，计算 `epsilon = rho * g_GC / (norm(g_GC) + 1e-12)`。
4. 保存参数副本，施加 `w + epsilon`，清除第一次梯度。
5. 重放第一次计算前的Python、NumPy、Torch CPU/CUDA RNG状态，对同一sample计算第二次损失和梯度。
6. 第二次梯度不做GC。使用 `copy_` 精确恢复原始参数，再按原阈值裁剪第二次梯度。
7. 基础AdamW只执行一次step；完整更新成功后，外层训练循环只更新一次EMA。

两次前向后将RNG恢复到第一次前向/反向结束时的状态，使随机流只推进一次。
Region采样和增强参数生成仍在原Dataset中，每次WSI访问只执行一次；训练函数不再次索取Dataset样本。
SR=0时沿用原分类专用编码分支，不读取HR或调用SR融合/解码。
冻结BN沿用原模型行为：固定running统计，不额外改变原有参数的requires_grad设置。

**与公开示例的明确区别：** 官方源码在GC前计算范数，论文公式(7)在GC后归一化。
本版遵循用户明确要求，采用GC后的范数。arXiv伪代码的descent符号也有歧义；
本版按正文、IEEE最终版及官方代码的ascent-only实现，第二次梯度不做GC。
权重恢复使用原值副本，避免 `w += epsilon; w -= epsilon` 的浮点残差。
基础优化器按任务要求使用AdamW，而官方示例使用Adam；保留原AdamW参数组和decay排除策略，
不额外添加或重复缩放L2罚项。不包含ASAM的权重自适应缩放。

## 配置和输出

完整配置为 `configs/native512_gcsam.yaml`，以五折MeanMIL配置为基础：
预训练ResNet18、仅layer4微调、冻结BN、pooling=[mean]、Top-K=5、全部Dropout=0、
VIB关闭、SR=0、整批编码、最多24个Region、BF16、增强、label smoothing=0.1、
encoder LR=1e-5、head LR=2e-4、weight decay=0.01、EMA=0.999、Plateau和早停均保留。
从epoch1到结束全程GCSAM；`gcsam.rho`默认0.05。新入口拒绝延后start_epoch、开启VIB、SR或Dropout。

每张WSI和history记录optimizer、base_optimizer、rho、两次分类损失、GC前/后及第二次梯度范数、
encoder/head学习率。原 `train_loss_cls` 和 `train_loss_total` 记录未扰动的第一次损失，
实际AdamW更新使用第二次梯度。`grad_norm`保留原共享编码器梯度范数的含义，测量发生在裁剪前。

Scheduler绑定真实的基础AdamW，每epoch沿用原验证loss调度。阈值、最优epoch选择和评估函数
直接复用原实现。checkpoint沿用原模型format、architecture和state_dict；optimizer_state保存
真实AdamW动量与参数组，并增加gcsam元数据，另外保存scheduler_state。
原 `eval_e2e.py`、`eval_native512_reference.py` 及模型加载器可直接读取新checkpoint。
入口仍要求新的或空的输出目录，未新增训练resume CLI。

## 本地测试

```bash
python -B -m unittest tests.test_gcsam -v
python -B -m unittest discover -s tests -v
```

测试包括独立数值方向oracle、多维GC和一维排除、扰动范数、逐值恢复、第二次梯度不做GC、
仅一次AdamW更新、先恢复后裁剪、冻结参数/BN统计、27取24且只采样一次、真实PIL增强像素重放、
Python/NumPy/CPU/CUDA RNG重放、BF16、NaN/Inf及零方向、AdamW动量恢复和后续更新、
rho=0与原AdamW逐值对照、EMA每WSI一次、Scheduler每epoch一次，以及原评估入口读取checkpoint。
测试使用随机/小尺寸合成样本，不代表正式五折效果或任意WSI规模的显存已经验证。

2026-11-04本机验收：全套150项测试通过（原139项 + 新增11项），耗时61.596秒，
包含CUDA BF16。另在RTX 3060上对随机 `[2,3,512,512]`、256维embedding、BF16执行
一个完整GCSAM更新：两次编码、一次AdamW step、一次EMA更新，参数变化且梯度有限。
开发前记录的412个已跟踪文件SHA-256全部保持一致，Git HEAD未改变；
`git diff --name-status` 无输出，`git status --short` 只有本次新增GCSAM文件。
未启动真实数据训练，未提交或推送Git。合成检查关闭预训练下载，正式YAML仍启用预训练权重。

## Fold 0–4 独立启动

只提供命令，本次开发不执行正式训练。服务器先同步新增文件，并激活原myconda环境。
以下路径沿用现有运行记录；每条命令独立执行，输出使用新目录。

```bash
python -u -B /root/E2E_MetaSR/train_e2e_gcsam.py --config /root/E2E_MetaSR/configs/native512_gcsam.yaml --data-root /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train --labels-csv /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/camelyon16_labels.csv --split-dir /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/native512_264 --split-scheme 5fold --fold 0 --output /root/E2E_MetaSR/runs/gcsam_mean_sr0_fold0_$(date +%Y%m%d_%H%M%S) --device cuda:0

python -u -B /root/E2E_MetaSR/train_e2e_gcsam.py --config /root/E2E_MetaSR/configs/native512_gcsam.yaml --data-root /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train --labels-csv /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/camelyon16_labels.csv --split-dir /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/native512_264 --split-scheme 5fold --fold 1 --output /root/E2E_MetaSR/runs/gcsam_mean_sr0_fold1_$(date +%Y%m%d_%H%M%S) --device cuda:0

python -u -B /root/E2E_MetaSR/train_e2e_gcsam.py --config /root/E2E_MetaSR/configs/native512_gcsam.yaml --data-root /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train --labels-csv /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/camelyon16_labels.csv --split-dir /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/native512_264 --split-scheme 5fold --fold 2 --output /root/E2E_MetaSR/runs/gcsam_mean_sr0_fold2_$(date +%Y%m%d_%H%M%S) --device cuda:0

python -u -B /root/E2E_MetaSR/train_e2e_gcsam.py --config /root/E2E_MetaSR/configs/native512_gcsam.yaml --data-root /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train --labels-csv /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/camelyon16_labels.csv --split-dir /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/native512_264 --split-scheme 5fold --fold 3 --output /root/E2E_MetaSR/runs/gcsam_mean_sr0_fold3_$(date +%Y%m%d_%H%M%S) --device cuda:0

python -u -B /root/E2E_MetaSR/train_e2e_gcsam.py --config /root/E2E_MetaSR/configs/native512_gcsam.yaml --data-root /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train --labels-csv /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/camelyon16_labels.csv --split-dir /mnt/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/downstream_train/native512_264 --split-scheme 5fold --fold 4 --output /root/E2E_MetaSR/runs/gcsam_mean_sr0_fold4_$(date +%Y%m%d_%H%M%S) --device cuda:0
```
