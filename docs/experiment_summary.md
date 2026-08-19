# RGB-D YOLO11-seg 实验总结

本文档总结当前 RGB-D 包裹实例分割项目的关键实验，说明为什么最终选择 `coord_attv2-s8/weights/best.pt` 作为学习和交付版本。

## 1. 任务背景

本项目基于 Ultralytics YOLO11-seg，面向 RGB-D 包裹实例分割和除双相关任务。

输入：

```text
RGB 图像 + depth 图像
```

输出：

```text
包裹实例 box
包裹实例 mask
类别预测
```

当前数据集：

```text
train: 13575 对 RGB-depth-label
val:   2669 对 RGB-depth-label
nc:    2
```

类别：

```text
0: 00
1: 01
```

从已有结果看，类别 0 表现较强，类别 1 是主要短板。因此评估模型时不能只看 `all`，也要重点看类别 1 的指标。

## 2. 最终版本

最终选定版本：

```text
coord_attv2-s8/weights/best.pt
```

完整路径：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

对应结构：

```text
YOLO11-seg + RGB-D 双分支 + CoordAttV2 融合 + P3 wavelet 引导 + Depth FPN
```

推荐名称：

```text
YOLO11-seg-RGBD-CoordAttV2-P3Wavelet-DepthFPN
```

对应日志：

```text
/workspace/ultralytics-main_for_genye/LOG/genye11s_just_coord_att_v2+p3+dfpn.log
```

对应结果目录：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8
```

## 3. 三组关键实验

当前最重要的三组实验如下。

| run              | 日志                                     | 结构                                | 作用                                    |
| ---------------- | ---------------------------------------- | ----------------------------------- | --------------------------------------- |
| `coord_attv2-s3` | `genye11s.log`                           | CoordAttV2 + P3 wavelet             | 旧主版本                                |
| `coord_attv2-s7` | `genye11s_just_coord_att_v2.log`         | just CoordAttV2                     | 去掉 P3 wavelet 和 Depth FPN 的轻量对照 |
| `coord_attv2-s8` | `genye11s_just_coord_att_v2+p3+dfpn.log` | CoordAttV2 + P3 wavelet + Depth FPN | 当前最终版本                            |

三组实验都基于同一数据集和同一任务，核心差异在 RGB-D 融合增强模块。

## 4. 配置对比

### 4.1 `s3`: CoordAttV2 + P3 wavelet

主要配置：

```yaml
rgbd_fusion: "coord_att_v2"
p3_wavelet_guided: True
depth_fpn: False
```

说明：

- 使用 CoordAttV2 做 RGB-D 融合。
- 在 P3 层加入 wavelet 引导。
- 不使用 Depth FPN。

### 4.2 `s7`: just CoordAttV2

主要配置：

```yaml
rgbd_fusion: "coord_att_v2"
p3_wavelet_guided: False
depth_fpn: False
```

说明：

- 只使用 CoordAttV2。
- 不加 P3 wavelet。
- 不加 Depth FPN。
- 该版本更轻，适合作为部署速度压力较大时的备选。

### 4.3 `s8`: CoordAttV2 + P3 wavelet + Depth FPN

主要配置：

```yaml
rgbd_fusion: "coord_att_v2"
p3_wavelet_guided: True
depth_fpn: True
```

说明：

- 使用 CoordAttV2 做基础 RGB-D 融合。
- 在 P3 层使用 wavelet 引导，增强小目标/边缘细节。
- 在 depth 分支加入轻量 Depth FPN，使 depth 多尺度特征先融合，再与 RGB 融合。
- 当前分割指标最强，因此作为最终版本。

## 5. 全量验证指标

以下结果来自完整验证集：

```text
/workspace/Datasets/final/images/val
```

验证集规模：

```text
2669 images
7246 instances
```

### 5.1 best epoch 指标

这里的 best 指标从 `results.csv` 中统计得到，主要用于观察训练过程中达到过的最好性能。

| run  | epochs | best Mask mAP50-95 | best epoch | best Mask P | best Mask R | best Mask mAP50 | best Box mAP50-95 |
| ---- | -----: | -----------------: | ---------: | ----------: | ----------: | --------------: | ----------------: |
| `s3` |    300 |            0.79678 |        262 |     0.92501 |     0.86700 |         0.91260 |           0.82519 |
| `s7` |    233 |            0.79749 |        232 |     0.92452 |     0.86983 |         0.91708 |           0.82502 |
| `s8` |    300 |        **0.80303** |        282 |     0.92320 |     0.87057 |         0.91696 |           0.82416 |

结论：

```text
s8 的 best Mask mAP50-95 最高，达到 0.80303。
```

### 5.2 final best.pt 验证指标

训练结束后，Ultralytics 会加载 `weights/best.pt` 再做一次验证。这里更接近交付时实际使用的权重表现。

| run  |       Box P |       Box R |   Box mAP50 | Box mAP50-95 |      Mask P |      Mask R |  Mask mAP50 | Mask mAP50-95 |
| ---- | ----------: | ----------: | ----------: | -----------: | ----------: | ----------: | ----------: | ------------: |
| `s3` |     0.92081 |     0.87092 |     0.91401 |      0.82316 |     0.92396 |     0.86941 |     0.91369 |       0.79603 |
| `s7` |     0.91627 | **0.87953** | **0.91802** |  **0.82420** |     0.92267 | **0.87613** | **0.92021** |       0.79475 |
| `s8` | **0.92149** |     0.86962 |     0.91533 |      0.82339 | **0.92633** |     0.87319 |     0.91975 |   **0.79863** |

结论：

```text
s8 的 final best.pt Mask mAP50-95 最高，适合作为最终分割版本。
s7 的 Box mAP50-95 和 Mask mAP50 略高，且模型更轻，可作为部署备选。
```

本任务以实例分割为主，因此优先选择 `Mask mAP50-95` 更高的 `s8`。

## 6. 类别 1 指标

类别 1 是当前主要短板，因此单独列出。

### 6.1 final best.pt 类别 1 指标

| run  | 类别1 Box P | 类别1 Box R | 类别1 Box mAP50 | 类别1 Box mAP50-95 | 类别1 Mask P | 类别1 Mask R | 类别1 Mask mAP50 | 类别1 Mask mAP50-95 |
| ---- | ----------: | ----------: | --------------: | -----------------: | -----------: | -----------: | ---------------: | ------------------: |
| `s3` |     0.89349 |     0.76177 |         0.84367 |            0.69990 |      0.90238 |      0.76355 |          0.84472 |             0.65903 |
| `s7` |     0.88226 | **0.77881** |     **0.85046** |            0.70059 |      0.89660 |  **0.77792** |          0.85610 |             0.66383 |
| `s8` | **0.89215** |     0.76027 |         0.84655 |        **0.70218** |  **0.90432** |      0.77037 |      **0.85647** |         **0.66651** |

结论：

```text
s8 在类别 1 的 Mask mAP50 和 Mask mAP50-95 上最高。
s7 在类别 1 recall 上更高。
```

如果后续业务更重视类别 1 不漏检，可以进一步比较 `s7` 和 `s8` 的可视化结果，并针对类别 1 调整阈值或做数据增强。

## 7. 模型规模和速度考虑

三组模型规模大致如下：

| run  | 结构                                |    参数量 | GFLOPs | 备注     |
| ---- | ----------------------------------- | --------: | -----: | -------- |
| `s3` | CoordAttV2 + P3 wavelet             | 约 16.97M |   51.9 | 旧主版本 |
| `s7` | just CoordAttV2                     | 约 16.11M |   49.2 | 更轻     |
| `s8` | CoordAttV2 + P3 wavelet + Depth FPN | 约 17.29M |   54.3 | 精度最优 |

结论：

```text
s8 精度最强，但计算量比 s7 更高。
如果部署平台对速度非常敏感，可以保留 s7 作为轻量备选。
```

## 8. 为什么最终选择 s8

选择 `s8/best.pt` 的原因：

1. 在完整验证集上，`best Mask mAP50-95` 最高，为 `0.80303`。
2. 在最终 `best.pt` 验证中，`Mask mAP50-95` 最高，为 `0.79863`。
3. 类别 1 的 `Mask mAP50-95` 最高，为 `0.66651`。
4. 模型规模只比 `s7` 略大，但分割指标更好。
5. 当前任务以实例分割为核心，优先看 mask 指标而不是 box 指标。

最终推荐：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

## 9. 注意事项

### 9.1 不要把 quick_start 指标当作正式指标

`examples/quick_start` 中只有：

```text
80 train
20 val
```

这是给新同学做 smoke test 的小数据包，只用于确认流程能跑通。

例如 `quick_val.py` 在 20 张 val 上得到的较高指标不能作为正式实验结果引用。

正式指标必须来自完整验证集：

```text
/workspace/Datasets/final/images/val
```

### 9.2 不要混用 s3/s7/s8 权重

当前统一约定：

```text
最终版 = s8/best.pt
轻量备选 = s7/best.pt
旧主版本 = s3/best.pt
```

写文档、发权重、做部署测试时要明确版本，避免混淆。

### 9.3 best.pt 优先于 last.pt

交付、验证、对比时优先使用：

```text
weights/best.pt
```

不要默认使用 `last.pt`，因为最后一个 epoch 不一定是验证集最优。

### 9.4 类别 1 仍需重点分析

虽然 `s8` 是当前最强版本，但类别 1 与类别 0 差距仍然明显。

后续优化建议：

- 统计类别 1 样本数量。
- 分析类别 1 尺度分布。
- 检查类别 1 漏标和边界质量。
- 看 `val_batch*_pred.jpg` 中类别 1 的失败样例。
- 尝试类别 1 重采样或针对性增强。
- 根据业务需求单独调类别 1 置信度阈值。

## 10. 给新同学的结论

如果只是学习和复现，按下面顺序即可：

1. 先跑 `examples/quick_start/quick_val.py`。
2. 再跑 `examples/quick_start/quick_train.py`。
3. 阅读 `README_RGBD_S8.md`。
4. 阅读 `docs/code_guide.md`。
5. 最后阅读本文档理解为什么选择 `s8`。

如果要继续做实验，建议从以下两个方向开始：

1. 以 `s8` 为基线，针对类别 1 做数据和阈值优化。
2. 以 `s7` 为轻量基线，评估部署速度和精度折中。

当前正式推荐版本：

```text
coord_attv2-s8/weights/best.pt
```
