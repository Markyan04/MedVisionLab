# APTOS 2019：ResNet50 × MECS VersionB × DAST 的 10-seed 消融

## 实验设计

脚本运行以下四组严格配对的实验：

| 实验名 | layer3 后的 MECS | 损失函数 |
| --- | --- | --- |
| `resnet_baseline` | 无 | Cross Entropy |
| `resnet_layer3_mesc` | VersionB | Cross Entropy |
| `resnet_dast` | 无 | DAST |
| `resnet_layer3_mesc_dast` | VersionB | DAST |

默认 seeds 为 `0 1 2 3 4 5 6 7 8 9`。四组使用完全相同的数据划分、图像增强、输入尺寸、训练预算、优化器、学习率计划和早停规则。同一 seed 的模型初始化、样本顺序和增强随机性相互配对。最优模型只按验证集 QWK 选择，测试集只在选择完成后评估。

为保证 10-seed 结果可比，建议所有实验固定同一台机器、同一 CUDA/PyTorch 版本。脚本启用了 cuDNN 确定性设置；但 VersionB 使用的 CUDA median/max-pool backward 在极少数数值并列情形下可能仍不是逐 bit 确定，因此统计结论应以 10-seed 的均值和样本标准差为准。

启动器会在 Python 进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，满足 CUDA 10.2 及以上环境中 PyTorch 确定性矩阵运算的要求。该设置约增加 24 MiB GPU workspace；如果直接运行 Python 训练入口，脚本也会自动设置相同默认值。

默认配置沿用当前 APTOS runner：256×256、batch size 32、最多 40 epochs、patience 10、AdamW、OneCycleLR、预训练 ResNet50。预训练骨干的最大学习率为 `1e-4`，新建的分类头和 MECS VersionB 参数为 `1e-3`。DAST 的两组统一使用论文默认值 `tau=1.0, gamma=1.5`。

## AutoDL 上运行

假设项目位于：

```text
/root/autodl-tmp/MedVisionLab
```

你的数据目录为 `/root/autodl-tmp/APTOS2019`，启动器会自动识别。训练标签既可以叫 `train_1.csv`（Kaggle 原文件名），也可以叫 `train.csv`。启动器还兼容项目内的 `APTOS-2019`/`APTOS2019` 和 `/root/autodl-tmp/APTOS-2019`；若位置不同，通过 `APTOS_DATA_ROOT` 指定。

AutoDL 的 PyTorch 镜像通常已有 `torch` 和 `torchvision`。其余依赖缺失时，在项目根目录执行：

```bash
pip install numpy pandas pillow scikit-learn
```

先做四组单 batch 检查（不训练、不保存 checkpoint）：

```bash
cd /root/autodl-tmp/MedVisionLab

for exp in resnet_baseline resnet_layer3_mesc resnet_dast resnet_layer3_mesc_dast; do
  python -u APTOS/aptos_10seed_ablation.py train \
    --experiment "$exp" \
    --seed 0 \
    --data-root /root/autodl-tmp/APTOS2019 \
    --output-root /root/autodl-tmp/aptos_10seed_ablation \
    --dry-run
done
```

随后运行完整实验：

```bash
cd /root/autodl-tmp/MedVisionLab
chmod +x APTOS/run_10seed_autodl.sh
export APTOS_DATA_ROOT=/root/autodl-tmp/APTOS2019
export APTOS_OUTPUT_ROOT=/root/autodl-tmp/aptos_10seed_ablation
bash APTOS/run_10seed_autodl.sh
```

建议在 `tmux` 中运行。也可以用 `nohup`：

```bash
mkdir -p /root/autodl-tmp/aptos_10seed_ablation
nohup bash APTOS/run_10seed_autodl.sh \
  >/root/autodl-tmp/aptos_10seed_ablation/nohup.log 2>&1 &
```

若显存不足，可统一降低四组的 batch size：

```bash
bash APTOS/run_10seed_autodl.sh --batch-size 16
```

额外参数会原样传给每一个训练进程。不要只对其中一组改变训练参数，否则不再是严格控制的消融。

## 中断续跑与选择部分实验

再次执行同一条命令即可续跑。已生成 `summary.json` 的实验会跳过；未完成的实验从 `last.pt` 的下一 epoch 恢复。恢复时会检查关键超参数，防止把不同配置接到同一个 checkpoint 上。

只测试少量 seeds：

```bash
APTOS_SEEDS="0 1" bash APTOS/run_10seed_autodl.sh
```

只运行部分实验：

```bash
APTOS_EXPERIMENTS="resnet_baseline resnet_layer3_mesc" \
  bash APTOS/run_10seed_autodl.sh
```

## 输出

每组每个 seed 位于：

```text
<output-root>/<experiment>/seed_<seed>/
├── best.pt
├── history.csv
├── run.log
└── summary.json
```

`last.pt` 是带优化器状态的中断续跑文件，仅在实验未完成时保留；成功生成 `summary.json` 后会自动删除以节省约十余 GB 的总磁盘占用。若确实要保留每组的 `last.pt`，启动时添加 `--keep-last`。`best.pt` 始终保留。

全部结束后自动生成：

- `runs.csv`：每个 seed 的结果；
- `aggregate.csv`：各实验的均值和样本标准差；
- `aggregate.md`：可以直接阅读/复制到实验记录中的 `mean ± std` 表；
- `launcher.log`：整个批量任务日志。

如果批量任务没有跑到汇总步骤，可手动执行：

```bash
python APTOS/aptos_10seed_ablation.py summarize \
  --output-root /root/autodl-tmp/aptos_10seed_ablation \
  --expected-seeds 10
```
