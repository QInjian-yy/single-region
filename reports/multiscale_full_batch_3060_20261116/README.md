# 多尺度整批与 RTX 3060 显存实测

本机 NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB 显存；PyTorch 2.13.0+cu130。
当前多尺度 Mean 的 N=5、16、32、64 均通过完整训练更新；Attention 的 N=64
以及 Mean 的 N=64、microbatch=1 对照也通过。每项独立启动一个 Python 进程，
执行 3 次实际 `train_wsi` 加 EMA 更新，共 18 次反传、AdamW 和 EMA 更新。

## 测试设置与边界

- 直接调用本地多尺度训练入口的 `train_wsi`，未修改模型和训练函数。
- LR 为 `[N,3,512,512]`；HR 为 `[N,2,3,256,256]`；embedding_dim=256。
- BF16 autocast，encoder 仅 layer4 可训练，freeze_bn=true，AdamW，EMA 配置为 0.999。
- SR 权重为 0；Mean 关闭 AEM，Attention 使用其原配置的 AEM。
- 使用合成 RGB 像素替代图像读取。当前机器没有 ResNet18 预训练权重缓存，
  因此仅将 `pretrained` 设为 false；架构、训练层和 tensor 尺寸保持原配置。
- 这是输入尺寸一致的合成训练步测试，不是实际数据训练，也不验证数据清单、
  图像配对或分类性能。显存结论限于以上配置及实测的 N。

## 整批证据

在 N=64 时，每个训练步的实际编码器输入只有两次：

```text
LR encoder 输入：[64,3,512,512]
LR layer4 输出：[64,512,16,16]
HR encoder 输入：[128,3,256,256]
HR layer4 输出：[128,512,8,8]
WSI 池化输入：[64,256]
```

其中 HR 输入由 `[64,2,3,256,256]` 展开而来。每步只调用一次
`encode_multiscale`、一次 WSI 池化、一次 backward 和一次 AdamW。
stage4 输出为 BF16 且带梯度；两路 stage4 及融合向量均收到反传梯度，
fusion 参数在更新后发生变化。

分块对照 N=64、microbatch=1 则每步有 64 次 `encode_multiscale`、
128 次 encoder 调用，最后仍统一池化并只更新一次。
第 2、3 步的 PyTorch 峰值已分配显存相同，未出现测量记录累积保留输入的问题。

## 显存峰值

单位为 GiB（1024³ bytes），取每项 3 步中的最大值。

| 模式 | N | PyTorch 张量峰值 | PyTorch 预留峰值 | 整卡采样峰值 |
|---|---:|---:|---:|---:|
| Mean 整批 | 5 | 0.3052 | 0.3730 | 2.2871 |
| Mean 整批 | 16 | 0.5767 | 0.6641 | 2.5781 |
| Mean 整批 | 32 | 0.9743 | 1.2031 | 3.1172 |
| Mean 整批 | 64 | 1.7712 | 2.2812 | 4.1953 |
| Attention 整批 | 64 | 1.7717 | 2.2832 | 4.1992 |
| Mean 分块，每次 1 region | 64 | 1.6002 | 1.7148 | 3.6289 |

PyTorch 预留值包含已分配给张量的空间，不能与张量峰值相加。
整卡数值使用 `nvidia-smi` 每 100 ms 采样，包含桌面和其他程序；
每项开始前整卡已占用约 1.7725 GiB。结束后恢复至 1815 MiB。

## 原始记录与复现

`summary.json` 为最终汇总。各项同名 JSON 包含逐步输入 shape、梯度回调、
前向/反传/优化器/EMA 各阶段显存、完整配置及训练源码 SHA256；
`.gpu.csv` 为整卡采样，`.log` 为执行日志。6 项记录的源码 SHA256 均与本地一致。

在项目根目录运行以下命令可重测一项，输出文件应使用新名字：

```powershell
& 'D:/conda/envs/clam_latest/python.exe' -B reports/multiscale_full_batch_3060_20261116/probe.py --n 64 --pooling mean --output reports/multiscale_full_batch_3060_20261116/recheck_mean_n64.json
```

`discarded_mock_retention/` 在本地保存已弃用的初测，未纳入 Git。初测的 Mock 调用历史会保留
CUDA 输入、抬高显存峰值，不用于本报告结论；最终探针使用普通函数替换，
只保存 shape 和数值，不保留 GPU 输入张量。
