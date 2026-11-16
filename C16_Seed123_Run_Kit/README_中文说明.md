# CAMELYON16 独立五折协议 · Seed 123

这是单组新的标准外层五折划分：393 WSI，233 正常、160 肿瘤，每张 WSI 在 5 次外层 Test 中恰好出现一次。Train/Val/Test 大约 3:1:1。固定随机种子 123，按照原始来源 × 类别分组随机打散后按预设配额均匀分到五个外层部分；**没有查看模型得分，不按容易预测筛选 Test**。

协议文件由使用者另行准备，JSON 含所有源清单 SHA256；`camil_attn_adamw_seed123.yaml` 是 Attention + AdamW 基线配置，没有开启 Top-K/ASRD/AEM/SR。训练随机种子仍为 1。

## 安装并启动（服务器 E2E_MetaSR_GCSAM 根目录）

准备好协议目录后，在仓库根目录运行：

```bash
conda activate clam_latest
C16_PROTOCOL_DIR=/path/to/camil_5fold_seed123_sourcestrat \
bash C16_Seed123_Run_Kit/launch.sh
```

默认协议位置是 `data_protocols/camil_5fold_seed123_sourcestrat`，可以用
`C16_PROTOCOL_DIR` 指向另行维护的目录。`C16_CONFIG` 可选择配置文件，默认读取
本目录的 `camil_attn_adamw_seed123.yaml`；`TRAIN_PYTHON` 可选择 Python 解释器。
启动器先检查 393 WSI、各折、配置和源 manifest 的 SHA256，通过后使用 `screen`
启动五折，并输出 `RUN`、`LOG` 和会话名。已有实验输出路径会触发拒绝覆盖检查。

## 查看结果

```bash
ls -t logs/*seed123*.log | head -n 1
tail -n 40 "$(ls -t logs/*seed123*.log | head -n 1)"
grep -E 'Test AUC=|five folds|TRAIN_EXIT_CODE' "$(ls -t logs/*seed123*.log | head -n 1)"
```

注意：原协议只使用 `slide_id` 作为分组代理；没有经过真实性验证的患者 ID，不能宣称患者级完全独立。划分不是提高 AUC 的模型创新；如果用 Test AUC 反复选择随机种子，分数会偏乐观，不能作为无偏最终论文指标。需要真正的患者 ID 对照再做患者级分组。原官方 test 已经并入 393 WSI，因此不能再当作独立的外部测试集。
