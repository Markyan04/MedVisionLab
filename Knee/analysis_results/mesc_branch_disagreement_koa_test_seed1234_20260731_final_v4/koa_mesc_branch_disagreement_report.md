# KOA MESC 分支差异验证报告

## 设置

- checkpoint：`E:\Paper\Computer-Vision-for-Medicine\Knee\checkpoints\best_resnet50_mecs_layer3_knee_oa_controlled_koa_resnet50_plus_mesc_seed1234.pt`
- checkpoint 最佳验证 QWK：0.8161
- 固定测试集：`E:\Paper\Computer-Vision-for-Medicine\Knee_Osteoarthritis\test`（n=1656）
- 模型：ResNet50 + layer3 MESC + CE（未重新训练、未修改 loss）
- 距离定义：三个 pairwise L1 距离均除以 `2*C`；三者之和为 `total_disagreement`。
- 四个等人数分组按 disagreement 升序；并列值使用 sample_id 的 SHA-256 确定性打散，避免 ImageFolder 的类别/路径顺序污染分组。
- 默认 forward 与可选返回路径的最大 logit 绝对差：0

## 四分位结果

| 四分位组 | n | Accuracy | 错误率 | MAE | 平均 disagreement |
|---|---:|---:|---:|---:|---:|
| Q1_low | 414 | 70.05% | 29.95% | 0.340580 | 0.00000000 |
| Q2 | 414 | 68.12% | 31.88% | 0.384058 | 0.00000000 |
| Q3 | 414 | 69.32% | 30.68% | 0.352657 | 0.00000000 |
| Q4_high | 414 | 67.63% | 32.37% | 0.381643 | 0.00000071 |

## 总体统计

- Accuracy：68.78%
- MAE：0.364734
- disagreement 唯一值数量：2
- disagreement 精确为 0：1655/1656（99.94%）
- disagreement 与 `abs(pred-label)` 的 Spearman rho：-0.016402（p=0.504772）
- 正确样本平均 disagreement：0.00000026
- 错误样本平均 disagreement：0.00000000
- Q4-Q1 错误率差：2.42 个百分点；风险比：1.081
- Q4 vs Q1 错误率 Fisher 精确检验：p=0.499511
- avg-max / avg-median / max-median 注意力元素精确相等率：99.94% / 100.00% / 99.94%

## 结论

- disagreement 高的样本是否更容易预测错误：当前结果不支持；disagreement 大量并列为 0，无法形成可辨识的高低梯度。
- disagreement 是否与序数误差正相关：否；观测相关不显著且指标退化。
- 最高四分位错误率是否明显高于最低四分位：否；四分位主要由 0 值并列打散产生。
- 是否值得继续尝试 adaptive DAST：不建议仅依据本次 post-sigmoid disagreement 结果推进；应先解决或重新定义可辨识的分支差异指标。

> 这里的 disagreement 仅表示 MESC 三个通道注意力分支的内部差异，不解释为临床不确定性。
