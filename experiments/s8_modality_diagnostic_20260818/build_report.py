from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, ndigits: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_table(payload: dict | None) -> str:
    if not payload:
        return "| 模式 | λ_R | λ_D | Box mAP50-95 | Mask mAP50-95 | Box mAP50 | Mask mAP50 |\n|---|---:|---:|---:|---:|---:|---:|\n| 未运行四态验证 | - | - | - | - | - | - |"
    rows = ["| 模式 | λ_R | λ_D | Box mAP50-95 | Mask mAP50-95 | Box mAP50 | Mask mAP50 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ["RD", "R-only", "D-only", "Empty"]:
        r = payload["modes"][mode]
        rows.append(
            f"| {mode} | {_fmt(r['rgb_lambda'], 1)} | {_fmt(r['depth_lambda'], 1)} | "
            f"{_fmt(r['box_map'])} | {_fmt(r['mask_map'])} | {_fmt(r['box_map50'])} | {_fmt(r['mask_map50'])} |"
        )
    return "\n".join(rows)


def _shapley_table(payload: dict | None) -> str:
    if not payload:
        return "| 指标 | RGB Shapley | Depth Shapley | RGB 占比 | Depth 占比 | 交互项 I_RD |\n|---|---:|---:|---:|---:|---:|\n| 未运行四态验证 | - | - | - | - | - |"
    name = {
        "box_map": "Box mAP50-95",
        "box_map50": "Box mAP50",
        "mask_map": "Mask mAP50-95",
        "mask_map50": "Mask mAP50",
    }
    rows = ["| 指标 | RGB Shapley | Depth Shapley | RGB 占比 | Depth 占比 | 交互项 I_RD |", "|---|---:|---:|---:|---:|---:|"]
    for r in payload["shapley"]:
        rows.append(
            f"| {name.get(r['metric'], r['metric'])} | {_fmt(r.get('rgb_contribution'))} | "
            f"{_fmt(r.get('depth_contribution'))} | {_fmt((r.get('rgb_ratio') or 0) * 100, 2) if r.get('rgb_ratio') is not None else 'N/A'}% | "
            f"{_fmt((r.get('depth_ratio') or 0) * 100, 2) if r.get('depth_ratio') is not None else 'N/A'}% | "
            f"{_fmt(r.get('interaction'))} |"
        )
    return "\n".join(rows)


def _script_detail_text(payload: dict | None) -> str:
    fusion_indices = payload.get("fusion_indices", [36, 37, 38]) if payload else [36, 37, 38]
    return f"""本实验主要由三个脚本完成：

| 脚本 | 作用 | 关键输出 |
|---|---|---|
| `run_four_state_val.py` | 加载 S8 checkpoint，在内存中 hook 三个 RGB-D 融合层，依次执行 RD/R-only/D-only/Empty 四态验证，并计算 Shapley 贡献 | `four_state_metrics.json`、`four_state_metrics.csv`、`shapley_metrics.csv`、四态柱状图、Shapley 堆叠图 |
| `plot_training_curves.py` | 读取原始 S8 训练目录中的 `results.csv`，绘制训练过程中的 mAP 与 loss 曲线 | `s8_training_map_curves.png`、`s8_training_loss_curves.png`、`training_curve_summary.json` |
| `build_report.py` | 读取上述 JSON/CSV 结果，自动生成本报告 | `REPORT.md` |

`run_four_state_val.py` 没有改项目源码，而是对已经加载到内存里的模型做临时 monkey patch。具体步骤如下：

1. 通过 `YOLO(weights)` 加载 `coord_attv2-s8/weights/best.pt`。
2. 定位 S8 的三个融合模块，当前 checkpoint 对应融合层索引为 `{fusion_indices}`，分别对应 P3、P4、P5 的 RGB-D fusion。
3. 保存每个融合模块原始的 `forward(rgb, depth)` 到 `_modality_original_forward`。
4. 用新的 `scaled_forward(rgb, depth)` 替换当前进程内的 `forward`：

```python
return original_forward(rgb * lambda_rgb, depth * lambda_depth)
```

5. 每个模式验证前设置一次 `lambda_rgb/lambda_depth`，然后调用原项目已有的 `YOLO.val()` 和 `SegmentationValidator` 计算 Box/Mask 指标。
6. 四态结果写入 CSV/JSON 后，脚本根据公式计算 RGB/Depth Shapley contribution 和交互项 `I_RD`。

另外，当前源码比旧 S8 checkpoint 多了 `learnable_blend/base_scale/out_scale` 等属性。脚本只在内存里给旧 checkpoint 补默认值：

```text
adaptive_gate = False
gate_mode = channel
base_scale = 1.0
out_scale = 1.0
learnable_blend = False
```

这些默认值等价于原始 S8 的普通 `rgb + out` 融合路径，不引入新的训练模块或新 loss。"""


def _parameter_table(payload: dict | None) -> str:
    if not payload:
        return "| 参数 | 值 | 说明 |\n|---|---|---|\n| 未运行四态验证 | - | - |"
    return f"""| 参数 | 值 | 说明 |
|---|---:|---|
| weights | `{payload['weights']}` | S8 `best.pt` checkpoint |
| data | `{payload['data']}` | 本次四态验证使用的数据配置 |
| split | `val` | 只在验证集上诊断，不训练 |
| imgsz | `{payload['imgsz']}` | 与原始训练保持 640 输入尺寸 |
| batch | `{payload['batch']}` | RTX 5060 Laptop GPU 上稳定运行 |
| device | `{payload['device']}` | 使用本机 CUDA GPU |
| workers | `{payload['workers']}` | Windows 下设为 0，避免多进程 dataloader 问题 |
| conf | `{payload['conf']}` | Ultralytics val 默认低置信度阈值，用于 mAP 统计 |
| iou | `{payload['iou']}` | NMS IoU 阈值 |
| feature mute layers | `{payload['fusion_indices']}` | P3/P4/P5 三个 RGB-D 融合入口 |
| λ_R/λ_D | `RD=(1,1), R-only=(1,0), D-only=(0,1), Empty=(0,0)` | Shapley 四态输入开关 |"""


def _result_analysis(payload: dict | None) -> str:
    if not payload:
        return "四态验证尚未运行，暂不能进行结果分析。"
    modes = payload["modes"]
    shp = {r["metric"]: r for r in payload["shapley"]}
    rd_box = modes["RD"]["box_map"]
    r_box = modes["R-only"]["box_map"]
    d_box = modes["D-only"]["box_map"]
    rd_mask = modes["RD"]["mask_map"]
    r_mask = modes["R-only"]["mask_map"]
    d_mask = modes["D-only"]["mask_map"]
    box_gain_r = rd_box - r_box
    mask_gain_r = rd_mask - r_mask
    box_gain_d = rd_box - d_box
    mask_gain_d = rd_mask - d_mask
    return f"""从四态验证看，S8 在当前验证集上明显更依赖 RGB，但 Depth 不是完全无效。

首先看单模态支撑能力。正常 RD 的 Box mAP50-95 为 {rd_box:.4f}，Mask mAP50-95 为 {rd_mask:.4f}。关闭 Depth 后，R-only 仍有 Box {r_box:.4f}、Mask {r_mask:.4f}，分别保留了 RD 性能的 {r_box / rd_box * 100:.2f}% 和 {r_mask / rd_mask * 100:.2f}%。关闭 RGB 后，D-only 只有 Box {d_box:.4f}、Mask {d_mask:.4f}，分别为 RD 的 {d_box / rd_box * 100:.2f}% 和 {d_mask / rd_mask * 100:.2f}%。这说明在该 checkpoint 和该验证集上，RGB 分支单独已经能支撑较多预测，Depth 分支单独支撑能力较弱，尤其对 mask 精细分割更弱。

再看边际增益。如果从 R-only 加回 Depth，Box mAP50-95 提升 {box_gain_r:.4f}，Mask mAP50-95 提升 {mask_gain_r:.4f}。这说明 Depth 对最终 RD 结果仍然有补充作用，并不是完全被模型忽略。相反，如果从 D-only 加回 RGB，Box mAP50-95 提升 {box_gain_d:.4f}，Mask mAP50-95 提升 {mask_gain_d:.4f}，提升更大，说明 RGB 对联合预测的主导性更强。

Shapley 分解给出更稳健的贡献率。对 Box mAP50-95，RGB 贡献为 {shp['box_map']['rgb_contribution']:.4f}，占 {shp['box_map']['rgb_ratio'] * 100:.2f}%；Depth 贡献为 {shp['box_map']['depth_contribution']:.4f}，占 {shp['box_map']['depth_ratio'] * 100:.2f}%。对 Mask mAP50-95，RGB 贡献为 {shp['mask_map']['rgb_contribution']:.4f}，占 {shp['mask_map']['rgb_ratio'] * 100:.2f}%；Depth 贡献为 {shp['mask_map']['depth_contribution']:.4f}，占 {shp['mask_map']['depth_ratio'] * 100:.2f}%。因此，如果只用一句话概括当前 S8：Box 任务约为 RGB 67% / Depth 33%，Mask 任务约为 RGB 75% / Depth 25%。

交互项方面，Box mAP50-95 的 `I_RD={shp['box_map']['interaction']:.4f}`，为负值，表示两模态在 Box 严格指标上存在一定冗余或融合干扰；Mask mAP50-95 的 `I_RD={shp['mask_map']['interaction']:.4f}`，接近 0 且略为正，说明 Depth 对 mask 严格指标有少量互补，但互补幅度很小。Box mAP50 的交互项为 {shp['box_map50']['interaction']:.4f}，负值更明显，说明在较宽松 IoU 阈值下 RGB 与 Depth 的检测信息重叠较多。

需要强调的是，这个结果还不能直接推出“Depth 没学好”。它只能证明：在联合训练后的 S8 checkpoint 中，把 RGB 静默后，Depth 路径能独立贡献的有效预测较少。要判断原因，还需要同等训练设置下的 Depth-only 单模态参考模型。如果 `Q_D^solo` 明显高于这里的 `Q_D^joint-mute`，才说明 Depth 在联合训练中可能被 RGB 压制；如果 `Q_D^solo` 本身也低，则更可能是当前 Depth 数据或 Depth 分支表达能力有限。"""


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    out_dir = exp_dir / "outputs"
    training = _read_json(out_dir / "training_curve_summary.json")
    four_state = _read_json(out_dir / "four_state_metrics.json")

    training_text = "尚未生成训练曲线摘要。"
    if training:
        training_text = (
            f"原始 S8 训练日志共 {training['epochs']} 个 epoch。Box mAP50-95 最优为 "
            f"{training['best_box_map']:.4f}，出现在 epoch {training['best_box_epoch']}；"
            f"Mask mAP50-95 最优为 {training['best_mask_map']:.4f}，出现在 epoch {training['best_mask_epoch']}。"
            f"最终 epoch {training['final_epoch']} 的 Box/Mask mAP50-95 分别为 "
            f"{training['final_box_map']:.4f}/{training['final_mask_map']:.4f}。"
        )

    val_params = "尚未运行四态验证。"
    dataset_note = (
        "四态验证尚未运行。"
    )
    if four_state:
        val_params = (
            f"权重：`{four_state['weights']}`；数据：`{four_state['data']}`；"
            f"imgsz={four_state['imgsz']}，batch={four_state['batch']}，device={four_state['device']}，"
            f"workers={four_state['workers']}，conf={four_state['conf']}，iou={four_state['iou']}；"
            f"feature-level mute 插入在融合层索引 `{four_state['fusion_indices']}`。"
        )
        dataset_note = (
            "注意：`coord_attv2-s8/args.yaml` 中原始训练数据为 `data-ubuntu-genye.yaml`，类别数 `nc=2`，"
            "路径指向 `/workspace/Datasets/final`。当前 Windows 工作区没有找到该原始 2 类数据集副本，"
            "因此本次四态验证使用 `Dataset/xinjiang_1500` 的本地 3 类验证集。"
            "所以四态验证的绝对 mAP 不应与原始训练日志里的 0.80 左右 mAP 直接横向比较；"
            "本报告更强调同一 checkpoint、同一验证集、同一评价设置下 RD/R-only/D-only/Empty 的相对差异与 Shapley 分解。"
        )

    report = f"""# S8 RGB-D 模态诊断实验报告

## 1. 实验目的

本实验把当前 S8 看作一个已经训练好的 RGB-Depth 多模态实例分割网络，先不修改 PMG，也不加入梯度调制。目标是回答三个问题：

1. RGB 和 Depth 在联合模型中分别能单独支撑多少检测/分割性能。
2. 当前 S8 的最终预测对 RGB、Depth 的 Shapley 贡献比例是多少。
3. 如果 Depth 贡献偏低，后续应优先判断是 Depth 信息不足，还是联合训练中被 RGB 压制。

## 2. 隔离方式

所有新增文件都放在 `{exp_dir}`，没有修改项目原始源码。为了便于复现，本目录保存了 `source_snapshot/`：包含本次实验参考的 `ultralytics/nn/tasks.py`、`ultralytics/models/yolo/segment/val.py`、S8 的 `args.yaml` 和 `results.csv` 快照。

四态验证不通过改源码实现，而是在 Python 进程内临时 hook S8 的三个 RGB-D 融合模块。S8 在 P3、P4、P5 三个尺度融合 RGB 与 Depth 特征，实验在融合入口处设置两个标量开关：

```text
F_R' = λ_R F_R
F_D' = λ_D F_D
Y = f(F_R', F_D')
```

四种模式为：

| 模式 | λ_R | λ_D | 含义 |
|---|---:|---:|---|
| RD | 1 | 1 | 正常 RGB+Depth |
| R-only | 1 | 0 | 只保留 RGB 特征，静默 Depth 特征 |
| D-only | 0 | 1 | 只保留 Depth 特征，静默 RGB 特征 |
| Empty | 0 | 0 | Shapley 基线 |

这里使用 feature-level mute，而不是把原始 depth 图片置零。原因是原始 0 值可能是有效/无效深度编码，且卷积、BN、bias 会让 raw zero 不等价于 feature zero。

## 3. 脚本具体做法

{_script_detail_text(four_state)}

## 4. 参数设定

{val_params}

{_parameter_table(four_state)}

原始训练配置来自 `coord_attv2-s8/args.yaml`：epochs=300，imgsz=640，batch=36，optimizer=MuSGD，lr0=0.01，lrf=0.001，momentum=0.937，weight_decay=0.0001，mosaic=1.0，fliplr=0.5，close_mosaic=10，overlap_mask=True。

{dataset_note}

## 5. 数学公式

令性能指标为 `v(R,D)`，例如 Box mAP50-95 或 Mask mAP50-95。四态验证得到：

```text
v(R,D) = RD 模式性能
v(R,0) = R-only 模式性能
v(0,D) = D-only 模式性能
v(0,0) = Empty 模式性能
```

RGB 的 Shapley 贡献定义为：

```text
C_R = 1/2 * [(v(R,0) - v(0,0)) + (v(R,D) - v(0,D))]
```

Depth 的 Shapley 贡献定义为：

```text
C_D = 1/2 * [(v(0,D) - v(0,0)) + (v(R,D) - v(R,0))]
```

二者满足效率性质：

```text
C_R + C_D = v(R,D) - v(0,0)
```

贡献率为：

```text
ρ_R = C_R / (v(R,D) - v(0,0))
ρ_D = C_D / (v(R,D) - v(0,0))
```

RGB-Depth 交互项定义为：

```text
I_RD = v(R,D) - v(R,0) - v(0,D) + v(0,0)
```

`I_RD > 0` 表示协同增益；`I_RD ≈ 0` 表示接近相加；`I_RD < 0` 常见于信息冗余或融合干扰。

以本次 Mask mAP50-95 为例，四态值为：

```text
v(R,D) = 0.313685
v(R,0) = 0.233229
v(0,D) = 0.075745
v(0,0) = 0
```

代入公式：

```text
C_R = 1/2 * [(0.233229 - 0) + (0.313685 - 0.075745)] = 0.235584
C_D = 1/2 * [(0.075745 - 0) + (0.313685 - 0.233229)] = 0.078101
ρ_R = 0.235584 / 0.313685 = 75.10%
ρ_D = 0.078101 / 0.313685 = 24.90%
I_RD = 0.313685 - 0.233229 - 0.075745 + 0 = 0.004711
```

## 6. 已生成结果

### 6.1 原始 S8 训练曲线

{training_text}

![S8 validation mAP curves](outputs/s8_training_map_curves.png)

![S8 training and validation losses](outputs/s8_training_loss_curves.png)

### 6.2 四态验证结果

{_mode_table(four_state)}

![Four-state mAP bars](outputs/four_state_map_bars.png)

### 6.3 Shapley 贡献结果

{_shapley_table(four_state)}

![Shapley contribution](outputs/shapley_contribution_stacked.png)

## 7. 结果分析

{_result_analysis(four_state)}

## 8. 如何复现

在项目根目录运行：

```powershell
& D:\\conda\\envs\\genye-yolo\\python.exe experiments\\s8_modality_diagnostic_20260818\\plot_training_curves.py
& D:\\conda\\envs\\genye-yolo\\python.exe experiments\\s8_modality_diagnostic_20260818\\run_four_state_val.py --device 0 --batch 8 --workers 0
& D:\\conda\\envs\\genye-yolo\\python.exe experiments\\s8_modality_diagnostic_20260818\\build_report.py
```

输出文件包括：

- `outputs/four_state_metrics.json`
- `outputs/four_state_metrics.csv`
- `outputs/shapley_metrics.csv`
- `outputs/s8_training_map_curves.png`
- `outputs/s8_training_loss_curves.png`
- `outputs/four_state_map_bars.png`
- `outputs/shapley_contribution_stacked.png`

## 9. 后续建议

如果 `D-only` 明显低于 `RD`，只能说明联合模型在关闭 RGB 后 Depth 支撑能力弱；还不能直接说明 Depth 没学好。要判断“Depth 本身没信息”还是“被 RGB 压制”，下一步应训练或收集同等设置下的 `S8_RGB-only` 与 `S8_Depth-only` 单模态参考模型，再比较：

```text
Gap_D = Q_D^solo - Q_D^joint-mute
Gap_R = Q_R^solo - Q_R^joint-mute
```

若 `Q_D^solo` 高而 `Q_D^joint-mute` 低，才更像联合训练中的 Depth 欠学习或模态竞争。训练过程中的梯度学习状态建议记录相对更新幅度：

```text
U_R(t) = ||η g_R(t)||_2 / (||Θ_R(t)||_2 + ε)
U_D(t) = ||η g_D(t)||_2 / (||Θ_D(t)||_2 + ε)
```

其中 `Θ_R/Θ_D` 分别为 RGB/Depth 分支参数，`g_R/g_D` 为对应梯度，`η` 为当前学习率，`ε` 取 `1e-12` 防止除零。
"""
    (exp_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(exp_dir / "REPORT.md")


if __name__ == "__main__":
    main()
