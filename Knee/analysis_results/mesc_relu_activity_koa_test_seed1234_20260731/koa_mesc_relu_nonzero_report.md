# KOA MESC ReLU 逐图非零比例

- checkpoint：`E:\Paper\Computer-Vision-for-Medicine\Knee\checkpoints\best_resnet50_mecs_layer3_knee_oa_controlled_koa_resnet50_plus_mesc_seed1234.pt`
- 最佳验证 QWK：0.8161450716583796
- test 样本数：1656
- 每个分支的 ReLU hidden shape：`[256,1,1]`，每张图共 256 个元素。
- 非零定义：`ReLU output > 0`。

| branch | 平均非零比例 | 标准差 | 最小值 | 最大值 | 全零图片数 | 存在非零元素图片数 |
|---|---:|---:|---:|---:|---:|---:|
| avg | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 1656 | 0 |
| max | 0.0000023588 | 0.0000959619 | 0.0000000000 | 0.0039062500 | 1655 | 1 |
| median | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 1656 | 0 |

> 这里统计的是 shared MLP 第一层后 ReLU 的激活比例，不是最终 attention 值。
