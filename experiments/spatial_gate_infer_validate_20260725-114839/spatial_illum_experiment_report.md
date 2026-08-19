# spatial_illum 模块有效性实验报告

## 1. 实验目的

本实验用于回答：RGB-D 融合网络中的 `spatial_illum` 门控模块，在过曝场景下是否真的起到了调节 RGB / Depth 权重的作用。

`spatial_illum` 位于 `CoordAttV2` 融合模块中。模型先将 RGB 特征和 Depth 特征拼接，再预测空间门控权重 `alpha`，融合形式为：

```text
base = alpha * rgb + (1 - alpha) * depth
```

其中：

- `alpha` 越大，表示越依赖 RGB；
- `1 - alpha` 越大，表示越依赖 Depth；
- `spatial_illum` 相比普通 spatial gate 额外输入了 RGB 的局部亮度均值和局部对比度；
- 当前配置 `rgbd_aux_loss=False`、`rgbd_gate_loss=False`，说明该 gate 没有显式监督，主要靠检测/分割任务损失自己学习。

因此，若模块有效，预期现象是：在局部过曝或低可靠 RGB 区域，`alpha` 应下降，Depth 权重应上升。

## 2. 实验位置和主要文件

实验目录：

```text
experiments/spatial_gate_infer_validate_20260725-114839
```

主要脚本：

| 文件                                 | 作用                                                                                                                    |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `infer_validate_spatial_gate.py`     | 加载模型和 RGB-D 数据，在推理过程中抓取每个 `CoordAttV2` 模块的 `_last_alpha`，导出 RGB/Depth 权重 CSV                  |
| `summarize_gate_effect.py`           | 按 normal / overexposed 图像分组，统计各层 `alpha`、Depth 权重、亮度、过曝像素比例及相关性                              |
| `analyze_local_overexposure_gate.py` | 根据亮度阈值自动生成局部过曝 mask，比较 mask 内外的 RGB/Depth 权重                                                      |
| `analyze_labeled_exposure_gate.py`   | 使用 `candidate_instance_manifest.csv` 中 selected=1 的实例标注，比较同一包裹在 normal 与 `_realexp` 版本中的 gate 变化 |

输出文件：

| 文件                                           | 内容                                          |
| ---------------------------------------------- | --------------------------------------------- |
| `outputs/spatial_gate_rgb_depth_weights.csv`   | 每张图、每个融合层的 `alpha` / Depth 权重统计 |
| `outputs/spatial_gate_summary_by_group.json`   | normal 与 overexposed 分组统计                |
| `outputs/local_overexposure_gate_summary.json` | 局部高亮区域内外的 gate 统计                  |
| `outputs/labeled_exposure_gate_summary.json`   | 标注包裹区域的成对 normal vs realexp 统计     |

## 3. 使用的模型和数据

模型配置：

```text
configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml
```

关键配置：

```yaml
rgbd_fusion: "coord_att_v2"
modality_adaptive_gate: True
gate_mode: "spatial_illum"
fusion_learnable_blend: True
rgbd_aux_loss: False
rgbd_gate_loss: False
```

权重：

```text
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt
```

数据：

```text
Dataset/xinjiang_1500_baoguang
split: val_ab
```

本次已保存结果中共检查：

- 图像对数量：547
- normal 图像：300
- overexposed 图像：247
- gate 统计行：1641
- 对应 3 个融合层：`model.36.coord_att`、`model.37.coord_att`、`model.38.coord_att`

三层的 gate map 尺度分别约为：

- `model.36.coord_att`: 80 x 80
- `model.37.coord_att`: 40 x 40
- `model.38.coord_att`: 20 x 20

## 4. 脚本具体怎么做实验

### 4.1 推理时导出 gate 权重

`infer_validate_spatial_gate.py` 的流程是：

1. 读取 RGB 图和对应 depth 图；
2. 对两者做相同的 letterbox 到 640 x 640；
3. 调用底层模型：

```python
raw_model(rgb_t, depth_t)
```

4. 遍历 `raw_model.named_modules()`；
5. 对每个带 `_last_alpha` 的模块，记录：

```text
src_weight_mean = alpha.mean()
depth_weight_mean = (1 - alpha).mean()
src_weight_min / max
depth_weight_min / max
base_blend_weight
out_blend_weight
```

6. 写入 `outputs/spatial_gate_rgb_depth_weights.csv`。

该脚本验证的是：模型中 `spatial_illum` gate 是否真的参与了前向推理，以及它在不同图像上的 RGB / Depth 权重分布。

### 4.2 normal / overexposed 整图分组统计

`summarize_gate_effect.py` 将文件名中带 `_realexp` 的图像视为 overexposed，否则视为 normal。

它统计：

- 每个融合层的平均 RGB 权重 `alpha`；
- 平均 Depth 权重 `1 - alpha`；
- 图像平均亮度；
- 亮度大于 0.88 的像素比例；
- normal 与 overexposed 的差值；
- 亮度指标与 `alpha` 的相关性；
- normal / `_realexp` 成对图像的权重变化。

这个实验回答的是：整张图变亮后，gate 的全局平均权重是否明显改变。

### 4.3 局部过曝区域 mask 分析

`analyze_local_overexposure_gate.py` 不只看整图平均，而是在 letterbox 后的 RGB 图上用灰度阈值生成过曝 mask：

```text
gray > 0.88
```

并过滤过小的 mask：

```text
min_mask_ratio = 0.005
```

然后对每个 gate map：

1. 将灰度图 resize 到 gate map 尺度；
2. 比较高亮 mask 内部与外部的平均 `alpha`；
3. 计算：

```text
inside_minus_outside_src = alpha_inside - alpha_outside
inside_minus_outside_depth = depth_inside - depth_outside
```

如果 `inside_minus_outside_src < 0`，说明高亮区域内 RGB 权重更低；如果 `inside_minus_outside_depth > 0`，说明高亮区域内 Depth 权重更高。

### 4.4 标注实例成对分析

`analyze_labeled_exposure_gate.py` 使用：

```text
Dataset/xinjiang_1500_baoguang/meta/candidate_instance_manifest.csv
```

其中 `selected=1` 的实例被认为是做过曝光增强的目标包裹。

脚本流程：

1. 找到同一 stem 的 normal 图和 `_realexp` 图；
2. 读取 YOLO label polygon；
3. 将 selected 实例合并成一个 selected mask；
4. 将非 selected 的包裹作为 control mask；
5. 将非包裹区域作为 background mask；
6. 分别在 normal 和 `_realexp` 上抓取每层 `alpha`；
7. 比较同一个 selected 包裹区域的变化：

```text
selected_src_delta = exposed_selected_src - normal_selected_src
selected_depth_delta = -selected_src_delta
```

该实验比整图统计更严格，因为它比较的是同一目标区域在增强前后的成对变化。

## 5. 主要结果

### 5.1 整图 normal / overexposed 统计

整图平均层面，overexposed 图像的过曝像素比例从 0.0245 增加到 0.0403，但三层 `alpha` 的平均变化都很小：

| 模块     | normal alpha | overexposed alpha | alpha 差值 | Depth 权重差值 |
| -------- | -----------: | ----------------: | ---------: | -------------: |
| model.36 |       0.9687 |            0.9698 |    +0.0012 |        -0.0012 |
| model.37 |       0.8557 |            0.8577 |    +0.0020 |        -0.0020 |
| model.38 |       0.7071 |            0.7082 |    +0.0011 |        -0.0011 |

成对图像统计中，只有 `model.38` 的平均变化略微符合预期：

| 模块     | 成对 alpha 差值均值 | Depth 差值均值 | Depth 增加比例 |
| -------- | ------------------: | -------------: | -------------: |
| model.36 |            +0.00075 |       -0.00075 |           0.8% |
| model.37 |            +0.00072 |       -0.00072 |          13.8% |
| model.38 |            -0.00026 |       +0.00026 |          50.6% |

结论：整图平均不适合证明 `spatial_illum` 有效。因为过曝通常是局部现象，整图平均会把局部响应稀释掉。

### 5.2 局部高亮 mask 内外比较

局部阈值 mask 分析显示，高亮区域内部的 `alpha` 普遍低于外部，即 RGB 权重下降、Depth 权重上升。

| 模块 / 分组            | 样本数 | mask 比例均值 | mask 内 alpha - mask 外 alpha | Depth 权重差值 | alpha 降低比例 |
| ---------------------- | -----: | ------------: | ----------------------------: | -------------: | -------------: |
| model.36 / normal      |    235 |        0.0166 |                       -0.0355 |        +0.0355 |          97.4% |
| model.37 / normal      |    172 |        0.0159 |                       -0.0252 |        +0.0252 |          64.0% |
| model.38 / normal      |    114 |        0.0147 |                       -0.0839 |        +0.0839 |          75.4% |
| model.36 / overexposed |    243 |        0.0255 |                       -0.0213 |        +0.0213 |          94.7% |
| model.37 / overexposed |    222 |        0.0224 |                       -0.0213 |        +0.0213 |          63.1% |
| model.38 / overexposed |    183 |        0.0184 |                       -0.1080 |        +0.1080 |          84.7% |

这是最能支持 `spatial_illum` 有效的结果。尤其是 `model.38.coord_att`，在高亮区域内的 RGB 权重下降最大，Depth 权重平均增加约 0.108。

### 5.3 标注过曝包裹区域成对比较

标注实例分析共使用：

- 成对 normal / `_realexp` 图像：247 对
- selected 过曝实例：450 个

结果如下：

| 模块     | selected alpha 差值 |             95% CI | selected Depth 差值 | 符合“Depth 增加”的比例 |
| -------- | ------------------: | -----------------: | ------------------: | ---------------------: |
| model.36 |             +0.0188 |   [0.0167, 0.0210] |             -0.0188 |                   6.5% |
| model.37 |             +0.0111 |   [0.0080, 0.0143] |             -0.0111 |                  29.1% |
| model.38 |             -0.0191 | [-0.0242, -0.0139] |             +0.0191 |                  71.7% |

差分中的差分结果也类似：

| 模块     | selected 相对 background 的 alpha 变化 | 结论                                       |
| -------- | -------------------------------------: | ------------------------------------------ |
| model.36 |                                +0.0187 | 与预期相反，过曝后 selected 区域更依赖 RGB |
| model.37 |                                +0.0108 | 与预期相反，但幅度小于 model.36            |
| model.38 |                                -0.0193 | 符合预期，过曝 selected 区域更依赖 Depth   |

结论：在语义包裹实例层面，`model.38.coord_att` 明确表现出预期效果；`model.36` 和 `model.37` 没有表现出预期的过曝抑制，甚至方向相反。

## 6. 结论：spatial_illum 是否有用

从已有实验看，`spatial_illum` 是有作用的，但作用并不均匀，不能简单说“三个尺度都稳定有效”。

可以支持的结论：

1. `spatial_illum` 确实在推理中生效，三个 `CoordAttV2` 融合层都产生了空间 `alpha` map。
2. 局部高亮区域内，RGB 权重普遍低于周围区域，Depth 权重普遍高于周围区域。这说明模块学到了一定的 RGB 可靠性判断。
3. 最深层 `model.38.coord_att` 的效果最符合设计目标。在自动高亮 mask 和标注 selected 实例两种分析中，它都倾向于降低过曝区域 RGB 权重、提高 Depth 权重。
4. 浅层 `model.36` 和中层 `model.37` 在标注实例成对实验中方向相反，说明它们可能更多保留纹理/边缘/局部外观信息，未必把过曝实例整体判断为“RGB 不可靠”。

不能直接支持的结论：

1. 现有 `experiments` 目录中的脚本没有完成严格的“有 spatial_illum vs 无 spatial_illum”同条件消融。
2. 因此，当前实验不能单独证明 `spatial_illum` 带来了 mAP 或业务指标提升。
3. 现有训练 CSV 可以作为背景指标，但这些结果混合了训练数据、过曝增强和模型配置差异，不能直接归因给 `spatial_illum` 模块本身。

最终判断：

```text
spatial_illum 对局部过曝区域的模态权重调节是有用的，尤其在 model.38 深层融合处证据较强；
但若要证明它提升最终检测/分割精度，还需要补做严格消融实验。
```

## 7. 建议补充的消融实验

为了更有力地证明模块是否提升最终效果，建议补做以下实验：

| 实验组                    | gate 配置                                                  | 目的                                          |
| ------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| Baseline                  | `modality_adaptive_gate=False`                             | 不使用自适应 RGB-D gate                       |
| Spatial                   | `modality_adaptive_gate=True`, `gate_mode='spatial'`       | 只使用空间 gate，不输入亮度/对比度先验        |
| Spatial Illum             | `modality_adaptive_gate=True`, `gate_mode='spatial_illum'` | 当前方法                                      |
| Spatial Illum + Gate Loss | `gate_mode='spatial_illum'`, `rgbd_gate_loss=True`         | 验证显式 gate 监督是否能纠正 P3/P4 的反向现象 |

所有实验应保持：

- 同一训练集；
- 同一验证集；
- 同一预训练权重；
- 同一 epoch、batch、imgsz、seed；
- 同一评估脚本和阈值。

建议报告指标：

- `mAP50(B)`、`mAP50-95(B)`；
- `mAP50(M)`、`mAP50-95(M)`；
- class 1 的 mask mAP；
- stack/business 指标；
- 过曝子集与 clean 子集分开评估；
- gate 行为统计继续保留，避免只看最终 mAP。

## 8. 一句话总结

当前脚本已经证明：`spatial_illum` 会在局部高亮区域动态降低 RGB 权重、提高 Depth 权重，深层 `model.38` 的证据最强；但还没有严格证明它带来了最终精度提升，需要补做同条件消融。
