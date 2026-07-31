# KOA MESC 分支塌缩定位报告

- checkpoint：`E:\Paper\Computer-Vision-for-Medicine\Knee\checkpoints\best_resnet50_mecs_layer3_knee_oa_controlled_koa_resnet50_plus_mesc_seed1234.pt`
- 最佳验证 QWK：0.8161450716583796
- 诊断 batch：10 张，每个 KL 等级 2 张，包含 `4/9215922R.png`。
- 仅推理诊断；未训练、未修改 loss、模型结构、checkpoint 或既有结果。

## 代码与运行时审计

- 返回值与手工分阶段计算逐元素一致：True
- 三个返回 tensor 的 data_ptr 全部不同：True
- 三个返回 Python tensor 对象不同：True
- 返回三分支之和与手工求和一致：True
- 默认 forward 与手工重构 logits 最大绝对差：0.00000000e+00
- 结论：没有发现变量复用、赋值错误、返回同一 tensor 或分析键映射错误。

## Batch 样本

| sample_id | label |
|---|---|
| 0/9003175L.png | 0 |
| 0/9003175R.png | 0 |
| 1/9001400L.png | 1 |
| 1/9001400R.png | 1 |
| 2/9003316R.png | 2 |
| 2/9006407R.png | 2 |
| 3/9011053L.png | 3 |
| 3/9012867L.png | 3 |
| 4/9012867R.png | 4 |
| 4/9215922R.png | 4 |

## 各阶段单分支统计

| stage | branch | shape | mean | std | min | max |
|---|---|---|---|---|---|---|
| raw_descriptor | avg | [10, 1024, 1, 1] | 3.57905120e-01 | 7.19278693e-01 | -1.46493167e-01 | 4.69394588e+00 |
| raw_descriptor | max | [10, 1024, 1, 1] | 2.36796498e+00 | 2.35288310e+00 | -4.05413620e-02 | 2.61758423e+01 |
| raw_descriptor | median | [10, 1024, 1, 1] | 2.54046470e-01 | 7.11651802e-01 | -1.59097955e-01 | 4.58548546e+00 |
| fc1_output | avg | [10, 256, 1, 1] | -1.82014930e+00 | 1.84734774e+00 | -1.13634291e+01 | -3.37584913e-01 |
| fc1_output | max | [10, 256, 1, 1] | -8.55320454e+00 | 9.55519009e+00 | -7.80004807e+01 | 3.97749841e-02 |
| fc1_output | median | [10, 256, 1, 1] | -1.49863589e+00 | 1.52043366e+00 | -9.07232952e+00 | -1.66830868e-01 |
| relu_hidden | avg | [10, 256, 1, 1] | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| relu_hidden | max | [10, 256, 1, 1] | 1.55371035e-05 | 7.85968616e-04 | 0.00000000e+00 | 3.97749841e-02 |
| relu_hidden | median | [10, 256, 1, 1] | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| pre_sigmoid_logits | avg | [10, 1024, 1, 1] | -4.05901065e-03 | 4.33990695e-02 | -1.50608182e-01 | 1.80551291e-01 |
| pre_sigmoid_logits | max | [10, 1024, 1, 1] | -4.05331375e-03 | 4.34062183e-02 | -1.50608182e-01 | 1.80551291e-01 |
| pre_sigmoid_logits | median | [10, 1024, 1, 1] | -4.05901065e-03 | 4.33990695e-02 | -1.50608182e-01 | 1.80551291e-01 |
| post_sigmoid_attention | avg | [10, 1024, 1, 1] | 4.98985380e-01 | 1.08445100e-02 | 4.62419003e-01 | 5.45015633e-01 |
| post_sigmoid_attention | max | [10, 1024, 1, 1] | 4.98986810e-01 | 1.08463028e-02 | 4.62419003e-01 | 5.45015633e-01 |
| post_sigmoid_attention | median | [10, 1024, 1, 1] | 4.98985380e-01 | 1.08445100e-02 | 4.62419003e-01 | 5.45015633e-01 |

## 各阶段分支间差异

| stage | pair | max_abs_diff | mean_abs_diff | exact_equal_fraction | same_data_ptr |
|---|---|---|---|---|---|
| raw_descriptor | avg-max | 2.50647812e+01 | 2.01005983e+00 | 0.00000000e+00 | False |
| raw_descriptor | avg-median | 1.12732267e+00 | 1.09617174e-01 | 0.00000000e+00 | False |
| raw_descriptor | max-median | 2.61921043e+01 | 2.11391830e+00 | 0.00000000e+00 | False |
| fc1_output | avg-max | 6.92963028e+01 | 6.73345661e+00 | 0.00000000e+00 | False |
| fc1_output | avg-median | 3.24565506e+00 | 3.22653830e-01 | 0.00000000e+00 | False |
| fc1_output | max-median | 7.23936768e+01 | 7.05502033e+00 | 0.00000000e+00 | False |
| relu_hidden | avg-max | 3.97749841e-02 | 1.55371035e-05 | 9.99609411e-01 | False |
| relu_hidden | avg-median | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | False |
| relu_hidden | max-median | 3.97749841e-02 | 1.55371035e-05 | 9.99609411e-01 | False |
| pre_sigmoid_logits | avg-max | 2.52168067e-03 | 1.21217767e-04 | 9.00000036e-01 | False |
| pre_sigmoid_logits | avg-median | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | False |
| pre_sigmoid_logits | max-median | 2.52168067e-03 | 1.21217767e-04 | 9.00000036e-01 | False |
| post_sigmoid_attention | avg-max | 6.30289316e-04 | 3.02905773e-05 | 9.00000036e-01 | False |
| post_sigmoid_attention | avg-median | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | False |
| post_sigmoid_attention | max-median | 6.30289316e-04 | 3.02905773e-05 | 9.00000036e-01 | False |

## data_ptr 检查

| stage | avg_data_ptr | max_data_ptr | median_data_ptr | all_data_ptr_distinct |
|---|---|---|---|---|
| raw_descriptor | 47527648768 | 47527689728 | 47645196288 | True |
| fc1_output | 47527730688 | 47527740928 | 47645237248 | True |
| relu_hidden | 47645247488 | 47645257728 | 47645267968 | True |
| pre_sigmoid_logits | 47645278208 | 47645319168 | 47645360128 | True |
| post_sigmoid_attention | 47645401088 | 47645442048 | 47645483008 | True |

## ReLU、pre-sigmoid 与 sigmoid 检查

| branch | relu_zero_fraction | pre_sigmoid_abs_gt_5_fraction | pre_sigmoid_abs_gt_10_fraction | pre_sigmoid_abs_gt_15_fraction | sigmoid_lt_1e-4_fraction | sigmoid_gt_1_minus_1e-4_fraction | sigmoid_extreme_fraction |
|---|---|---|---|---|---|---|---|
| avg | 1.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| max | 9.99609411e-01 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| median | 1.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |

## shared MLP 参数范数

| layer | parameter | shape | l2_norm | mean | std | min | max |
|---|---|---|---|---|---|---|---|
| fc1 | weight | [256, 1024, 1, 1] | 1.01462822e+01 | -2.01040902e-03 | 1.97147168e-02 | -1.64707869e-01 | 2.56103992e-01 |
| fc1 | bias | [256] | 3.12979132e-01 | -3.53978621e-03 | 1.92382503e-02 | -6.80465251e-02 | 2.96611246e-02 |
| fc2 | weight | [1024, 256, 1, 1] | 1.89593468e+01 | 2.22212839e-06 | 3.70299779e-02 | -1.30302325e-01 | 1.78726390e-01 |
| fc2 | bias | [1024] | 1.39483106e+00 | -4.05901019e-03 | 4.33990657e-02 | -1.50608182e-01 | 1.80551291e-01 |

## 分支直接替换测试

| setting | channel_attention_max_abs_diff | mecs_output_max_abs_diff | logits_max_abs_diff | logits_mean_abs_diff | logits_exact_equal_fraction | changed_prediction_count |
|---|---|---|---|---|---|---|
| original | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | 0 |
| median_replaced_by_avg | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | 0 |
| max_replaced_by_avg | 6.30259514e-04 | 4.15337086e-03 | 4.27246094e-04 | 2.74991980e-05 | 8.99999976e-01 | 0 |
| all_avg | 6.30259514e-04 | 4.15337086e-03 | 4.27246094e-04 | 2.74991980e-05 | 8.99999976e-01 | 0 |

## 最终定位

- 判定：**B. shared MLP/ReLU**
- 原因：Raw descriptors differ, but avg and median become exactly equal at the ReLU hidden stage and remain equal afterward.
- 原始描述符具有明显差异：True
- 三分支最小 ReLU 零元素比例：9.99609411e-01
- 三分支最大 sigmoid 极端值比例：0.00000000e+00
