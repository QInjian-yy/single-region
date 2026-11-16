# CAMELYON16 · 可选 LR+HR 多尺度 End-to-End + SR（Seed 123）

此包以**新增独立目录**方式提供训练代码：**不覆盖、不修改**原 `E2E_MetaSR_GCSAM` 中的 `models/`、`engine.py`、`train_e2e*`、已有 YAML、已有 `runs`、`data_protocols` 或历史 checkpoint。适配你当前安装过的 Spatial Top-K 版本，Top-K 在两组新配置中均保持**关闭**。

## 这两组是什么

| 组别 | 输入与梯度路径 | SR Loss | AEM | 其他 |
|---|---|---:|---|---|
| LR+HR E2E | LR 512 和 HR 256 局部像素 → **同一个可训练 ResNet18 layer4** → 特征融合 → Attention MIL → WSI CE | 0 | 开 | 不开 ASRD/Top-K |
| LR+HR E2E + SR | 与上面相同，同时 LR encoder stage1..4 → 原 SR decoder → 与对应 HR 像素比较 L1 | **0.05**，训练全程有效 | 开 | 不开 ASRD/Top-K |

**重要：这是分类端到端（HR 像素在线编码，没有预提取/冻结特征）与第二组的分类+SR 联合训练；但两组分类推理时也需要 HR 局部 Patch。它们不是“只用 LR 推理”的模型。** 若论文要求 LR-only 推理，下一阶段需做训练时 HR 教师、推理时 LR 学生的蒸馏，不能把当前方案说成 LR-only。

ResNet18 来自项目已使用的 ImageNet 预训练权重，**不使用 Phikon-v2**，不安装/升级额外 Python 包。LR 和 HR 共用一份 ResNet18，保持原始编码器 `layer4` 可训练，完整 WSI 分类损失的梯度回传到该层及新融合模块；SR Decoder 仅接收 LR 特征，SR 目标来自 HR 局部真值。

`HR Patch` 是原数据中 Level1 `images_8192/*.jpg` 的 **256×256 原始像素**，不把 8192 整图送进 GPU。为使每轮训练不会反复解码 8K JPEG，首次预处理只缓存 **每个 Region 2 张 PNG 像素小图** 和原 HR 坐标，不保存网络特征；这**不破坏端到端梯度**。Patch 由每张 Region 的 LR 图像上组织比例/局部颜色纹理、空间分散规则**独立且无标签**挑选，不依据肿瘤 XML、测试预测、测试 AUC。此策略只有局部采样，不保证涵盖稀疏转移灶，不能作为最终分辨率实验上限。

SR 开启时，每张 WSI 每轮从 Region 中抽最多 **2 个 Region** 参与 SR 辅助监督（每个 Region 一个 128×128 HR GT crop），用于控制 3090 24GB 显存；**其他所有 Region 仍参与分类损失**。SR `lambda=0.05` 不会在 3 轮后自动关闭。`ema_decay=0.999`、`min_epochs=5`、`patience=12`、分类头与 AEM 大体保持之前基线；设定最大 300 Epoch，早停可提前结束。

## 一键启动

把 `E2E_MultiScale_OptIn.tar.gz` 上传至项目根目录 `E2E_MetaSR_GCSAM`，在远程终端执行：

```bash
cd /media/ub/shidang_plu/MT/py_project/yujian/double_wsi/CAMELYON16_WSI_DATA/E2E_MetaSR_GCSAM
tar -xzf E2E_MultiScale_OptIn.tar.gz
bash E2E_MultiScale_OptIn/launch.sh
```

启动器**先在前台**运行 6 个 CPU smoke tests + Seed 123 数据与 HR 路径校验，通过后启动 **1 个 GNU screen**。这个 screen **顺序**执行：一次 HR 小 Patch 像素缓存预处理 → 实验1五折 → 实验2五折。防止并行显存不足，当前工程原代码和之前的 screen 均不会被修改。

你必须先具备：`clam_latest` Python 环境（PyTorch、torchvision、sklearn、PyYAML、Pillow 等原依赖）；原 Level1 `images_8192` 仍完整；已有独立 Seed 123 协议 `data_protocols/camil_5fold_seed123_sourcestrat`。首次解码 6531 个 8K HR JPEG 缓存约 **13,062** 张 256 PNG，可能耗费较长时间与若干 GB 磁盘；中断后再次运行新 session 将跳过已有 PNG；完成后复用缓存。

若 `screen` 不存在，请先安装 GNU screen（你的服务器之前已使用）。脚本固定使用 `/home/ub/anaconda3/envs/clam_latest/bin/python`，可用 `MULTISCALE_PYTHON` 环境变量指定其他 Python。用 `MULTISCALE_PROTOCOL` 可更换独立协议目录，但不要按 Test AUC 选择协议。

## 查看进度 / 结果

```bash
cd /media/ub/shidang_plu/MT/py_project/yujian/double_wsi/CAMELYON16_WSI_DATA/E2E_MetaSR_GCSAM
screen -ls
LOG=$(ls -t logs/s123_e2e_multiscale_*.log | head -n 1)
tail -n 40 "$LOG"
grep -E 'cache\]|EXPERIMENT|Test AUC=|five folds|TRAIN_EXIT_CODE|Traceback|CUDA out of memory' "$LOG"
```

正常全部结束最后应有 `TRAIN_EXIT_CODE=0`。实验目录自动创建：

- `runs/s123_e2e_multiscale_sr0_时间戳/`（每折 `history.csv`、`best.pth`、`best_val_predictions.csv`、`test_predictions.csv`、`test_metrics.json`，整体 `summary.json`）。
- `runs/s123_e2e_multiscale_sr005_时间戳/`（同样结构）。
- `E2E_MultiScale_OptIn/cache/lr512_hr256_k2_v1/`（只有缓存像素和无标签坐标索引）。

如果出现 CUDA OOM：保留日志，先退出/完成占用 GPU 的其他程序；不要直接修改原模型或强制覆盖结果。若单个 WSI 超大仍 OOM，可把**本 addon 两份 YAML**里的 `multiscale.patches_per_region` 从2改成1，同时必须使用**全新的**缓存目录；会得到另一组实验，不能与原 K2 混称。

如果再次启动，自动产生全新 `runs` 和日志，不覆盖旧结果；但未完成的五折**没有 checkpoint 续训功能**，只能选择重新运行。PNG 像素缓存可以继续复用。

## 研究与限制

- 该 addon 的 checkpoint 使用独立 `format = c16-e2e-lrhr-shared-resnet-sr-aem-v1`；原有 `eval_e2e.py` 不支持新格式，**也不受影响**。新训练入口每折自动用当前最优 Val AUC 模型评估 Test。
- 数据依然是当前混合的 393 WSI Seed 123 五折；标识 `slide_id_proxy_NO_PATIENT_LEVEL_GUARANTEE`，**没有证明患者级不交叉**。
- Seed 123 的 Test 已用于多轮选择，新的 Test AUC 仍属于探索性评估；正式论文需要预先固定协议、用独立且患者级隔离的数据做最终验证。
- 本地实现验证：6 项自动化测试通过（包括真实 online encoder 梯度、SR decoder 梯度、配对图像几何变换、缓存提取），已兼容原模型与先前安装的 Spatial Top-K opt-in 版本；未在你的 3090 上实际跑完 393 WSI 的五折，因此不能保证运行时长、性能收益或显存峰值。你服务器中的 `vendor` 未上传，本地测试仅以占位导入避免无关的旧 MetaRDN import 阻塞，未测试 MetaRDN 功能。
