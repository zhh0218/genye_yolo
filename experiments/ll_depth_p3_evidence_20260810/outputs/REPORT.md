# LL 作为 Depth P3 输入的证据

数据集：`Dataset\xinjiang_1500`，split：`val`  
样本数：`200`

## 关键结果

| 指标 | Mean | Median | 说明 |
|---|---:|---:|---|
| LL-depth Pearson r | 0.9572 | 0.9607 | LL 与原始 depth 的结构一致性。 |
| LL-depth PSNR | 23.38 dB | 23.73 dB | LL 作为平滑重构的保真度。 |
| LL energy share | 0.9860 | 0.9878 | Haar 系数中低频承载的能量比例。 |
| HF-boundary AUC | 0.6299 | 0.6268 | 高频幅值对 GT mask 边界的区分度。 |
| LL-gradient boundary AUC | 0.6642 | 0.6585 | LL 平滑后仍保留的边界梯度信号。 |

## 结论

当前 S8 的设计是合理的：`LL -> depth_low -> CoordAttV2` 给融合模块提供稳定的深度几何/主体结构；`LH/HL/HH -> edge_gate` 则把突变、边缘和局部细节留给单独的乘性增强分支。实验数据支持这种分工：LL 与原 depth 的相关均值为 `0.9572`，低频能量占比为 `0.9860`，说明 LL 不是丢掉 depth，而是在保留主体结构的同时去掉细碎高频。

因此，若问题是“LL 作为 CoordAttV2 的 depth 输入是否合适”，答案是：合适，尤其适合 CoordAtt 这种依赖全局/坐标方向池化的融合模块，因为它更稳定、更像几何先验；高频部分直接进入 CoordAtt 反而可能把噪声和无关深度突变混入主融合路径。

需要注意：这份实验是数据级/信号级证据，不等价于最终 AP 消融。若要论文级闭环，建议后续在同一训练设置下比较三个变体：`LL as CoordAtt input`、`raw depth as CoordAtt input + HF gate`、`HF/mixed depth as CoordAtt input`。

## 输出文件

- `s8_p3_wavelet_flow.png`：S8 P3 wavelet-guided CoordAttV2 结构图。
- `aggregate_charts.png`：总体统计图。
- `case_visualizations/`：单样本 Haar 分解可视化。
- `ll_depth_metrics.csv`：逐图指标。
- `summary.json`：聚合统计。
