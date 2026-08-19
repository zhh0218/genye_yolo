# S8 移植 COFNet GT-Mask 与对比学习：代码证据审计及改造报告

> 修订日期：2026-08-04  
> 唯一目标网络：`README_RGBD_S8.md` 定义的 `coord_attv2-s8`。  
> 目的：分析 COFNet 的 box-level mask 引导、伪掩码预测和对比学习能否移植到 S8，以改善只露出一角的小包裹、强曝光、暗色包裹等漏检问题。  
> 重要限制：本文只证明“代码当前做了什么”和“改造应接在哪里”；没有完成的新实验一律不写成有效果。

---

## 0. 结论与证据等级

### 0.1 本报告只认定以下 S8

本项目 README 对最终版本的原文是：

```text
# README_RGBD_S8.md:7-20
当前最终版本为：
YOLO11-seg + RGB-D 双分支 + CoordAttV2 融合 + P3 wavelet 引导 + Depth FPN

最终权重：
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

因此，本文分析基线固定为：

- `scale: 's'`；
- RGB 与 depth 双 backbone；
- P3 使用 wavelet-guided CoordAttV2；
- P4、P5 使用普通 CoordAttV2；
- depth 分支启用 `DepthLightFPN`；
- YOLO11 segmentation head 输出 P3/P4/P5；
- 数据为 README 记录的 `/workspace/Datasets/final`，2 类 `00/01`。

以下内容**不作为 S8 现状证据**：新疆 1500 三分类数据、`spatial_illum`、后续 reliability gate、STAL、ProgLoss、OL-IoU、`gatedfull` 日志及其他实验分支。

### 0.2 证据标签

本文使用四类标签：

- **[S8 已实现]**：能在 `README_RGBD_S8.md` 和当前 S8 路径代码中直接定位。
- **[COFNet 官方实现]**：能在作者公开仓库的实际训练代码中定位。
- **[待实现建议]**：尚未出现在 S8 中，只说明明确的接入位置，不声称有效。
- **[目前未知]**：仓库和 README 均不足以证明，必须通过原始数据或实验确认。

### 0.3 先给出经代码核验的关键结论

1. 原始 S8 确实是三尺度 RGB-D 融合，但有效 loss 只有 box、segmentation、classification；DFL 与 RGB-D auxiliary 均关闭。
2. S8 已经加载 segmentation GT mask，但当前 forward 只接收 RGB/depth，GT mask 没有进入 P3/P4/P5 融合。
3. COFNet 官方训练代码确实构造 box mask、训练 PMG，并执行模态特征与 mask 特征的对比 loss。
4. 但 COFNet 公开 `main` 分支中，将 mask 乘入 query 的 `q = q * x` 被注释；因此公开代码不能证明 GT mask 已实际改变 attention 输出。
5. 对 S8 最可核验的第一步是只在 P3 加 PMG 和 mask supervision；先证明它能预测包裹前景，再测试真正的 mask-guided fusion 和 depth-mask contrast。

---

## 1. S8 基线：结构证据

### 1.1 配置开关

**[S8 已实现]** `ultralytics/cfg/models/11/yolo11-seg.yaml:7-25`：

```yaml
scale: "s"
end2end: False
reg_max: 1
rgbd_fusion: "coord_att_v2"
p3_wavelet_guided: True
p3_wavelet_HF: False
p3_wavelet_adaptive: False
p4_wavelet_guided: False
depth_fpn: True
modality_adaptive_gate: False
rgbd_reliability_gate: False
cls01_weight: 1.0
ol_iou: False
prog_loss: False
stal: False
```

直接结论：原始 S8 开启的是 `coord_att_v2 + P3 wavelet + depth_fpn`；自适应模态门、RGB-D reliability gate、OL-IoU、ProgLoss 和 STAL 均关闭。

### 1.2 RGB/depth 双主干与分割头

**[S8 已实现]** `ultralytics/cfg/models/11/yolo11-seg.yaml:34-78`。下面只展示结构索引，`# ...` 明确表示省略未改写的中间层：

```yaml
backbone:
  # ... RGB P3/P4/P5 对应局部层 4/6/10

head:
  # ...
  - [[16, 19, 22], 1, Segment, [nc, 32, 256]] # P3, P4, P5

backboneD:
  # 与 RGB backbone 同布局
  # depth P3/P4/P5 同样对应局部层 4/6/10
```

**[S8 已实现]** `ultralytics/nn/tasks.py:220-267` 给出了真实前向路径：

```python
_depth_save_map = {4: None, 6: None, 10: None}

for did in range(_ds, _de):
    md = self.model[did]
    xd = md(xd)
    depth_local = did - _ds
    yd[depth_local] = xd

for rgb_stage in _depth_save_map:
    _depth_save_map[rgb_stage] = yd[rgb_stage]

# RGB backbone 只先运行 0..10
for m in self.model:
    if m.i > 10:
        break
    x = m(x)

if self.depth_fpn:
    p3d, p4d, p5d = depth_fpn_module(_depth_save_map[4], _depth_save_map[6], _depth_save_map[10])

_fusion_keys = [4, 6, 10]
for fi, key in enumerate(_fusion_keys):
    rgb_feat, depth_feat = y[key], _depth_save_map[key]
    fusion_module = self.model[self._fusion_indices[fi]]
    y[key] = fusion_module(rgb_feat, depth_feat)
```

所以 S8 确实在 backbone 的 P3/P4/P5 三个尺度融合，不是只在检测头末端拼接 depth。

### 1.3 P3、P4、P5 分别使用什么融合

**[S8 已实现]** `ultralytics/nn/tasks.py:2938-2986`：

```python
_fusion_key = d.get("rgbd_fusion", "se")
_fusion_cls = _RGBD_FUSION_CLS.get(_fusion_key, RGBDCrossAttention)
_p3_wavelet_guided = bool(d.get("p3_wavelet_guided", False))
_p4_wavelet_guided = bool(d.get("p4_wavelet_guided", False))

_wavelet_guided_cls = _WAVELET_GUIDED_MAP.get(_fusion_key)
_p4_fusion_cls = _wavelet_guided_cls if _p4_wavelet_guided and _wavelet_guided_cls else _fusion_cls

_p3_fusion_cls = _wavelet_guided_cls if _p3_wavelet_guided and _wavelet_guided_cls else _fusion_cls
_p3_module = _p3_fusion_cls(ch_p3, num_heads=8, kv_pool=10, **_fusion_kw)
_p4_module = _p4_fusion_cls(ch_p4, num_heads=8, kv_pool=10, **_fusion_kw)
_p5_module = _fusion_cls(ch_p5, num_heads=8, kv_pool=10, **_fusion_kw)
```

结合 YAML 的 `p3_wavelet_guided=True`、`p4_wavelet_guided=False`，可以确定：

| 尺度           | S8 实际模块                   | 代码依据                  |
| -------------- | ----------------------------- | ------------------------- |
| P3 / stride 8  | `RGBDWaveletGuidedCoordAttV2` | `tasks.py:2973-2984`      |
| P4 / stride 16 | `RGBDCoordAttV2`              | `tasks.py:2974,2982,2985` |
| P5 / stride 32 | `RGBDCoordAttV2`              | `tasks.py:2983,2986`      |

#### P3/P4/P5 stride 与感受野通俗说明

输入原图尺寸：**640×640**

- P3 stride=8 → 特征图：80×80；1 个特征像素，只看原图 8×8 的一小块区域
- P4 stride=16 → 特征图：40×40；1 个像素看原图 16×16
- P5 stride=32 → 特征图：20×20；1 个特征像素看原图 32×32 甚至更大一块区域

> ✅ **感受野**：特征图上某一个点，能覆盖原图多大范围。
> stride 越大，特征图虽然像素少，但是每一个像素管原图很大一片地方。

#### 为什么特征图边长变短，反而适合检测大物体？

假设图里有一个很大的包裹，占原图 200×200 像素。

- 在 P3 (80×80)：这个大包裹会横跨几十个特征像素。每个 P3 像素只看到包裹的一小块边角，看不到“整个包裹长什么样”，容易被局部纹理、深度噪声、过曝干扰。网络只看到碎片，很难识别这是一整个大物体。
- 在 P5 (20×20)：这个大包裹只占少数几个 P5 特征点。P5 的单个像素就能覆盖包裹的很大一部分，能看到物体整体轮廓、全局空间关系。虽然特征图本身分辨率低，没有精细边缘，但是它抓的是物体整体、大结构。

> 不是 P5“看到更多像素点”，是 P5 每一个点管的原图面积巨大，擅长捕获大目标的整体。

反过来小包裹，原图只有 24×24：

- P3：一个 P3 像素 8×8，小包裹会占好几个 P3 点，可以捕捉它的细节。
- P5：P5 一个点就覆盖 32×32，小包裹直接被吞进一个像素里面，细节全部丢失，分不清有没有小物体。

👉 所以小物体交给 P3。

##### YOLO 三层金字塔分工（通俗版）

- **P3 stride=8（80×80）细粒度**
  每个点看很小一块原图。擅长：小包裹、边缘、纹理、深度细节；不擅长大物体全局。

- **P4 stride=16（40×40）中等粒度**
  适配中等大小包裹。

- **P5 stride=32（20×20）粗粒度，大感受野**
  每个点覆盖原图很大区域。擅长：大包裹、远距离物体、全局空间；丢失细小细节。

> 概念区分
>
> - **分辨率**：特征图本身有多少个像素
> - **感受野**：这个像素映射回原图能看多大地盘
>   二者是完全不同概念，P5 不是分辨率高，而是感受野大。

### 1.4 P3 wavelet 到底做了什么

**[S8 已实现]** `ultralytics/nn/tasks.py:1925-1980`：

```python
ll, lh, hl, hh = self._haar_split(depth)
depth_low = F.interpolate(self.low_proj(ll), size=target_size, mode="nearest")
edge_feat = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
edge_gate = torch.sigmoid(F.interpolate(self.edge_gate(edge_feat), size=target_size, mode="nearest"))
base = self.coord_att(rgb, depth_low)
return base + self.beta * base * edge_gate
```

可证明的事实只有：P3 将 depth 做 Haar 分解；低频进入 CoordAttV2，高频生成单通道 `edge_gate` 对融合结果做乘性增强。

### 1.5 Depth FPN 到底做了什么

**[S8 已实现]** `ultralytics/nn/tasks.py:1987-2000`：

```python
def forward(self, p3, p4, p5):
    p4 = self.reduce5to4(torch.cat([self.up5(p5), p4], dim=1))
    p3 = self.reduce4to3(torch.cat([self.up4(p4), p3], dim=1))
    return p3, p4, p5
```

因此 Depth FPN 是 P5→P4→P3 的 top-down 聚合；它改善 depth 多尺度上下文，但代码中没有 depth 前景监督，也没有深度有效性/置信度监督。

### 1.6 原始 CoordAttV2 对光照问题的客观限制

**[S8 已实现]** YAML 关闭 `modality_adaptive_gate`，而 `ultralytics/nn/tasks.py:1845-1883` 的关闭分支为：

```python
x = torch.cat((rgb, depth), dim=1)
# coordinate attention
out = self.proj(x * a_w * a_h)

if self.adaptive_gate:
    # 此分支在原始 S8 中关闭
    base = alpha * rgb + (1 - alpha) * depth
    return self.base_scale * base + self.out_scale * out

return self.base_scale * rgb + self.out_scale * out
```

构造函数默认 `base_scale=1.0`、`out_scale=1.0`，原始 S8 也未覆盖这两个值。因此原始 S8 的有效形式是：

```text
fused = rgb + attention(concat(rgb, depth))
```

这只说明：S8 没有显式的 `alpha*RGB + (1-alpha)*depth` 模态权重，因而不能直接从门值观察曝光/过暗区域是否主动降低 RGB 权重。它**不能单独证明** depth 没有起作用。

---

## 2. S8 基线：数据与预处理证据

### 2.1 README 记录的数据，而不是其他实验数据

**[S8 已实现]** `README_RGBD_S8.md:49-65`：

```yaml
train: /workspace/Datasets/final/images/train
val: /workspace/Datasets/final/images/val
test: /workspace/Datasets/final/images/test

nc: 2
names:
  0: 00
  1: 01
```

当前 Windows 工作区没有 README 中的 `/workspace/Datasets/final` 原始文件，因此本文不能证明该数据的 depth 位深、毫米范围、无效值编码、图像分辨率分布或曝光样本数量。

### 2.2 RGB 与 depth 如何配对

**[S8 已实现]** `ultralytics/data/build.py:121-157`：

```python
depth_path = data.get(f"depth_{mode}")
if not depth_path and isinstance(img_path, str):
    candidate = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}depth{os.sep}")
    if candidate != img_path and Path(candidate).exists():
        depth_path = candidate
dataset = YOLORGBDDataset if depth_path else YOLOMultiModalDataset if multi_modal else YOLODataset
```

**[S8 已实现]** `ultralytics/data/dataset.py:345-368` 按文件主干名配对；缺少 depth 的 RGB 会被跳过，多余 depth 会告警跳过。

### 2.3 depth 的读取与通道形式

**[S8 已实现]** `ultralytics/data/dataset.py:372-417`：

```python
im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
if im is None:
    raise FileNotFoundError(f"Depth Image Not Found {f}")

if im.ndim == 2:
    im = np.repeat(im[..., None], 3, axis=2)
elif im.ndim == 3 and im.shape[2] == 1:
    im = np.repeat(im, 3, axis=2)

label["depth_img"] = depth_img
```

事实是：读取时保留原始 dtype，单通道 depth 被复制为 3 个相同通道。是否为 uint8/uint16 必须检查原始数据，不能由这段通用代码猜测。

### 2.4 训练与验证的 depth 归一化

**[S8 已实现]** `ultralytics/models/yolo/detect/train.py:113-126`：

```python
batch["img"] = batch["img"].float() / 255
depth_dtype = batch["depth_img"].dtype
depth = batch["depth_img"].float()
if depth_dtype == torch.uint8:
    batch["depth_img"] = depth / 255
else:
    depth_max = depth.amax(dim=(1, 2, 3), keepdim=True)
    batch["depth_img"] = depth / (depth_max + 1e-7)
```

`ultralytics/models/yolo/detect/val.py:80-94` 使用相同分支。因此：

- uint8 depth 固定除以 255；
- 非 uint8 depth 按每个 batch 样本自身最大值缩放。

这段代码可能改变跨样本绝对尺度，但在不知道原始 depth 的物理含义和无效值编码前，本文只把它列为审计点，不断言它一定导致漏检。

### 2.5 几何增强是否同步

**[S8 已实现]** 当前代码对 RGB/depth 同步执行：

| 增强                     | depth 证据位置                                        |
| ------------------------ | ----------------------------------------------------- |
| Mosaic                   | `ultralytics/data/augment.py:636-666,705-736,777-820` |
| MixUp                    | `ultralytics/data/augment.py:968-970`                 |
| RandomPerspective/Affine | `ultralytics/data/augment.py:1376-1421`               |
| 上下/左右翻转            | `ultralytics/data/augment.py:1637-1656`               |
| LetterBox                | `ultralytics/data/augment.py:1760-1826`               |
| 转 Tensor                | `ultralytics/data/augment.py:2289-2313`               |

同步几何变换已经存在，不能再写成“RGB 与 depth 一定错位”。但代码确实使用常数 `114` 填充 depth：

```python
# ultralytics/data/augment.py:1388-1393
border_value = (114, 114, 114) if depth_img.ndim == 3 and depth_img.shape[2] == 3 else 114
depth_img = cv2.warpPerspective(depth_img, M, dsize=self.size, borderValue=border_value)
```

`114` 对原始 S8 depth 是否合理目前未知；必须先统计原始 depth 的 dtype、有效范围和无效值，再决定是否改成 0、远平面值或附加 valid mask。

---

## 3. S8 当前真正启用的 Loss：逐行证据

### 3.1 criterion 确实是分割损失

**[S8 已实现]** `ultralytics/nn/tasks.py:719-723`：

```python
def init_criterion(self):
    """Initialize the loss criterion for the SegmentationModel."""
    stal = getattr(self, "stal", False)
    if getattr(self, "end2end", False):
        return E2ESegmentLoss(self, prog_loss=getattr(self, "prog_loss", False), stal=stal)
    return v8SegmentationLoss(self, stal=stal)
```

原始 S8 的 `end2end: False`，所以使用 `v8SegmentationLoss`。

### 3.2 主 loss 的四个槽位

**[S8 已实现]** `ultralytics/utils/loss.py:474-556`：

```python
loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl

loss[2] = self._classification_loss(pred_scores, target_scores, dtype, target_scores_sum)  # BCE

if fg_mask.sum():
    loss[0], loss[3] = self.bbox_loss(
        pred_distri,
        pred_bboxes,
        anchor_points,
        target_bboxes / stride_tensor,
        target_scores,
        target_scores_sum,
        fg_mask,
    )
    loss[1] = self.calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
    )

loss[0] *= self.hyp.box
loss[1] *= self.hyp.box
loss[2] *= self.hyp.cls
loss[3] *= self.hyp.dfl
loss[1] += self._rgbd_auxiliary_loss(batch)
return loss * batch_size, loss.detach()
```

分割单实例损失的核心也有直接代码，`ultralytics/utils/loss.py:579-581`：

```python
pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()
```

### 3.3 DFL 在 S8 中关闭

配置 `reg_max: 1`，而 `ultralytics/utils/loss.py:263-269` 为：

```python
self.reg_max = m.reg_max
self.use_dfl = m.reg_max > 1
```

所以 S8 的 DFL 项为 0 是配置决定的正常行为。

### 3.4 RGB-D 辅助 loss 并没有在原始 S8 启用

虽然当前 `loss.py` 中存在后来加入的 RGB-D gate/depth auxiliary 代码，但是否启用由总开关控制。`ultralytics/utils/loss.py:242-246`：

```python
self.rgbd_aux_loss = _bool_value(getattr(h, "rgbd_aux_loss", y.get("rgbd_aux_loss", False)))
self.rgbd_gate_loss = self.rgbd_aux_loss and _bool_value(getattr(h, "rgbd_gate_loss", y.get("rgbd_gate_loss", True)))
self.rgbd_depth_aux_loss = self.rgbd_aux_loss and _bool_value(
    getattr(h, "rgbd_depth_aux_loss", y.get("rgbd_depth_aux_loss", True))
)
```

原始 S8 YAML 没有 `rgbd_aux_loss` 字段，所以默认是 `False`。两个具体 loss 也有关闭保护：

```python
# ultralytics/utils/loss.py:429-453
if not (self.rgbd_gate_loss and self.rgbd_gate_loss_weight > 0 and alphas):
    return torch.zeros((), device=self.device)

if not (self.rgbd_depth_aux_loss and self.rgbd_depth_aux_loss_weight > 0 and logits is not None):
    return torch.zeros((), device=self.device)
```

模型构建同样只在开关打开时才加入 depth auxiliary head，`ultralytics/nn/tasks.py:2947-2948,2997-3001`：

```python
_rgbd_aux_loss = bool(d.get("rgbd_aux_loss", False))
_rgbd_depth_aux_loss = _rgbd_aux_loss and bool(d.get("rgbd_depth_aux_loss", True))

if _rgbd_depth_aux_loss:
    layers.append(RGBDDepthAuxHead(ch_p3))
```

因此，**原始 S8 的有效 loss 是 box + segmentation + classification，DFL 关闭，RGB-D auxiliary 为 0；不存在 COFNet 对比 loss，也不存在 COFNet 伪掩码 loss。**

---

## 4. COFNet 官方代码到底实现了什么

本节只引用作者官方仓库 [li554/COFNet](https://github.com/li554/COFNet)，不根据模块名称臆测。

### 4.1 box-level GT mask 的来源

**[COFNet 官方实现]** 训练入口先调用 `box2masks`：

```python
# train.py:329-337
imgs = imgs.to(device).float() / 255.0
imgs_rgb = imgs[:, :3, :, :]
imgs_ir = imgs[:, 3:, :, :]
cu_targets = targets.to(device)
target_masks = box2masks(cu_targets, imgs_rgb, device)
```

[`utils/general.py:56-72`](https://github.com/li554/COFNet/blob/main/utils/general.py#L56-L72) 将每个 GT box 的矩形范围填入 mask：

```python
target_masks = torch.zeros((b, 1, h, w), device=device)
xywh = targets[:, -4:] * torch.tensor([w, h, w, h], device=device)
# xywh -> tx1, ty1, tx2, ty2
for i in range(targets.shape[0]):
    target_masks[int(targets[i, 0]), 0, ty1[i] : ty2[i], tx1[i] : tx2[i]] = targets[i, 1] + 1
target_masks = target_masks.repeat(1, 3, 1, 1)
```

所以论文所说的 box-level mask 在官方训练代码中确实是由检测框栅格化得到，而不是精细实例轮廓。

### 4.2 PMG：从双模态特征预测伪掩码

**[COFNet 官方实现]** [`models/common.py:2138-2176`](https://github.com/li554/COFNet/blob/main/models/common.py#L2138-L2176)：

```python
class PMG(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.spatial_attention = SpatialAttention()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(c, c // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.LeakyReLU(),
            DeepWiseConv(c // 4, c // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.LeakyReLU(),
            nn.Conv2d(c // 4, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )


mask_f_conv = self.pmg(self.alpha * vi_f.detach() + self.beta * ir_f.detach())
```

注意 `vi_f.detach()` 和 `ir_f.detach()`：伪掩码预测支路的梯度不会通过这两个输入反向更新可见光/红外特征。

### 4.3 wrapper 训练时传 GT mask、推理时传预测 mask，但公开代码中 mask 未实际进入 attention 运算

同一官方模块 [`models/common.py:2168-2186`](https://github.com/li554/COFNet/blob/main/models/common.py#L2168-L2186)：

```python
if self.training:
    mask = F.interpolate(
        x3.mean(dim=1).unsqueeze(dim=1), size=(vi_f.size(2), vi_f.size(3)), mode="bilinear", align_corners=True
    )
    y = self.transformer(mask, ir_f, vi_f)
    return y, mask_f_conv
else:
    mask = torch.ones(mask_f_conv.size(), device=mask_f_conv.device).half()
    if p == "real" and x3 is not None:
        mask = F.interpolate(
            x3.mean(dim=1).unsqueeze(dim=1), size=(vi_f.size(2), vi_f.size(3)), mode="bilinear", align_corners=True
        )
    elif p == "predicted":
        mask = torch.sigmoid(mask_f_conv)
    y = self.transformer(mask, ir_f, vi_f)
    return y
```

到这一层只能证明：训练期把 GT box mask 传给 transformer，正常推理期把 PMG 预测 mask 传给 transformer。还不能据此断言 mask 真正改变了融合结果。

继续检查该模块实际导入的 [`models/conv_transformer.py:295-335`](https://github.com/li554/COFNet/blob/main/models/conv_transformer.py#L295-L335)：

```python
def forward(self, x, y, z):
    _b, _c, _h0, _w0 = x.shape
    x, y, z = self.avg_pool(x), self.avg_pool(y), self.avg_pool(z)
    q, k, v = self.q(y), self.k(z), self.v(z)
    x = rearrange(x, "b (head c) h w -> b head (h w) c", head=1)
    q = torch.nn.functional.normalize(q, dim=-1)
    # q = q * x
    k = torch.nn.functional.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature
    out = attn @ v
```

`x` 就是传入的 GT/预测 mask。它只被池化和 reshape；唯一把它乘到 query 的 `q = q * x` 被注释，后面的有效运算不再读取 `x`。因此，**按 COFNet 当前公开 `main` 分支执行，GT/预测 mask 没有实际参与 `NewCrossAttention` 的输出计算。**

这与论文/模块命名所表达的“mask-guided fusion”意图不一致。移植到 S8 时不能直接复制并声称已经得到 GT-mask attention；必须明确实现一条有效的 mask 运算，再做梯度和消融验证。也不能未经实验就假定“直接取消该注释”是正确修复。

### 4.4 对比学习使用哪些表示

**[COFNet 官方实现]** [`models/yolo_mask_v2.py:183-212`](https://github.com/li554/COFNet/blob/main/models/yolo_mask_v2.py#L183-L212) 保存可见光、红外、融合和 mask 特征，并做 pooling：

```python
vi_f = x[0]
ir_f = x[1]
mask_f = x[2].detach()
# ...
vi_pool = self.avg_pool(vi_f).view(len(vi_f), -1)
ir_pool = self.avg_pool(ir_f).view(len(ir_f), -1)
fu_pool = self.avg_pool(x).view(len(x), -1)
mask_pool = self.avg_pool(mask_f).view(len(mask_f), -1)
```

实际训练使用的并非泛化的 `models/loss.py::ClipLoss`，而是 [`utils/re_loss2.py:60-72`](https://github.com/li554/COFNet/blob/main/utils/re_loss2.py#L60-L72) 中的 `ContrastLoss`：

```python
mask_f = F.normalize(mask_pool, p=2, dim=1)
labels = torch.arange(len(mask_pool), device=mask_pool.device)
vi_f = F.normalize(feature_pool, p=2, dim=1)
vi_logits = torch.matmul(vi_f, mask_f.T) / 0.1
vi_loss_i = F.cross_entropy(vi_logits, labels)
vi_loss_t = F.cross_entropy(vi_logits.T, labels)
return (vi_loss_i + vi_loss_t) / 2
```

可证明的含义：同一个 batch 下，同索引的模态特征与 mask 特征为正匹配，其他索引为负匹配；温度在代码中固定为 `0.1`。

### 4.5 COFNet 的真实 loss 组合

**[COFNet 官方实现]** [`train.py:374-389`](https://github.com/li554/COFNet/blob/main/train.py#L374-L389)：

```python
pred, vi_pool, ir_pool, fu_pool, mask_pool, layer_maskfs, layer_maskf_conv = model(
    imgs_rgb, imgs_ir, target_masks, p=opt.res_mode
)
detection_loss, re_loss, loss_items = compute_loss(pred, cu_targets, imgs_rgb.shape[-2:])
mask_l = mask_loss(layer_maskf_conv, layer_maskfs, opt.loss_type)
sim_v_l = sim_loss(vi_pool, mask_pool)
sim_r_l = sim_loss(ir_pool, mask_pool)
loss = detection_loss + opt.lamb[1] * sim_v_l + opt.lamb[2] * sim_r_l
if opt.lamb[0] != 0 and opt.lamb[3] != 0:
    loss += opt.lamb[0] * re_loss + opt.lamb[3] * mask_l
```

官方默认参数位于 [`train.py:578-580`](https://github.com/li554/COFNet/blob/main/train.py#L578-L580)：

```python
parser.add_argument("--lamb", default=[1.0, 0, 0.15, 1.0], help="loss weights (re, sim_v, sim_r, mask)")
```

这说明官方默认关闭 visible-mask 对比项、保留 infrared-mask 对比项 `0.15`。该数值属于 COFNet 默认值，**不是 S8 depth 对比损失的已验证权重**。

还需注意：官方 `ComputeLoss` 虽然返回 `re_loss`，但 [`utils/re_loss2.py:417-449`](https://github.com/li554/COFNet/blob/main/utils/re_loss2.py#L417-L449) 中它初始化为 0，真正累加 reconstruction loss 的代码被注释：

```python
lcls, lbox, lobj, lrk, re_loss = (
    torch.zeros(1, device=device),
    torch.zeros(1, device=device),
    torch.zeros(1, device=device),
    torch.zeros(1, device=device),
    torch.zeros(1, device=device),
)
# re_loss += self.mse_loss(pmask, tmask)
```

所以当前公开代码中的 `re_loss` 有组合项但实际为 0；本文不把 reconstruction loss 列为已有效训练的 COFNet 模块。

---

## 5. COFNet 与 S8 的可迁移接口：只写有代码落点的方案

### 5.1 S8 已经有 GT 实例 mask，但目前没有送到融合层

**[S8 已实现]** `ultralytics/data/augment.py:2298-2306`：

```python
if self.return_mask:
    if nl:
        masks, instances, cls = self._format_segments(instances, cls, w, h)
        masks = torch.from_numpy(masks)
    labels["masks"] = masks
```

**[S8 已实现]** `ultralytics/nn/tasks.py:445-458`：

```python
depth = (
    depth_batch["img"]
    if isinstance(depth_batch, dict)
    else batch.get("depth_img")
    if depth_batch is None
    else depth_batch
)
preds = self.forward(batch["img"], depth) if preds is None else preds
return self.criterion(preds, batch)
```

S8 前向只接收 RGB 和 depth，GT mask 只在 criterion 中从 `batch` 读取。因此论文意图中的“训练期 GT mask 直接引导融合”不能只靠加一个 loss 完成，必须改 forward 数据接口，或采用预测 mask 始终引导、GT 只监督预测 mask 的变体。

### 5.2 建议 A：先移植 P3 单尺度伪掩码监督

**[待实现建议]** 最小范围只改 P3，依据是：

- P3 是 S8 的最高分辨率检测特征，配置明确为 stride 8：`yolo11-seg.yaml:55,65`；
- P3 当前融合调用点明确：`tasks.py:260-267` 的 `key=4`；
- P3 当前已经有 wavelet-guided fusion：`tasks.py:1971-1980`；
- S8 已经有 GT segmentation mask：`augment.py:2298-2306`。

建议的新增数据流（这是待实现设计，不是现有代码）：

```text
RGB P3 + Depth-FPN P3
        │
        ├─ 现有 RGBDWaveletGuidedCoordAttV2 ─→ fused P3
        │
        └─ 新增轻量 PMG ─→ mask_logits_P3
                              │
GT instance masks → union/downsample → mask supervision
```

第一步只监督 `mask_logits_P3`，暂不让 mask 改写融合结果。这样可以先回答一个可验证问题：P3 的 RGB-D 特征能否预测出包裹前景。没有该实验结果前，不能声称 mask attention 能改善小目标。

### 5.3 GT target 应优先使用 S8 的实例轮廓，而不是重新造矩形框

**[待实现建议]** S8 是 segmentation 网络，criterion 已经能从实例 mask 构造前景。当前仓库已有但在 S8 中未启用的辅助函数可作为实现参考，`ultralytics/utils/loss.py:385-409`：

```python
masks = batch.get("masks")
if masks.ndim == 3 and masks.shape[0] == b:
    obj = masks.gt(0).float().unsqueeze(1)
else:
    obj = torch.zeros((b, 1, *masks.shape[-2:]), device=self.device)
    batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
    for i in range(b):
        selected = masks[batch_idx == i]
        if selected.numel():
            obj[i, 0] = selected.gt(0).float().amax(dim=0)
obj = F.interpolate(obj, size=size, mode="nearest")
```

这段函数存在不等于原始 S8 已启用；第 3.4 节已经证明总开关关闭。迁移时可以复用其“实例 mask 求 union 并下采样”的逻辑。

如果实验要求严格复现 COFNet 的 box-level mask，再由 `batch["bboxes"]` 栅格化矩形；两种 target 不能混写，应作为两个消融版本。

### 5.4 建议 B：伪掩码验证通过后，再让 mask 引导 P3 融合

**[待实现建议]** 两种合法实现必须分开：

1. **按 COFNet wrapper 接口实现**：训练融合用 GT mask，推理融合用 `sigmoid(mask_logits)`；同时必须自己实现有效的 mask 运算。公开代码的 `q = q * x` 被注释，不能原样复制后声称 mask 已经引导 attention。
2. **一致路径变体**：训练和推理都用预测 mask，GT 只用于监督 `mask_logits`。这不是 COFNet 官方原式，只是减少训练/推理输入差异的 S8 改造候选。

建议先在 `tasks.py:260-267` 的 P3 `key=4` 处接入，P4/P5 暂不改。只有 P3 对照实验为正，才扩展 P4。报告不建议直接三尺度全改，因为目前没有 S8 实验证据证明三尺度均有收益。

### 5.5 建议 C：mask 分支成立后，再加 depth-mask 对比学习

**[待实现建议]** COFNet 默认只给 IR-mask 对比项非零权重；映射到 RGB-D 任务后，最直接的单变量实验是 `depth_P3 ↔ mask_feature_P3`，不是一开始同时约束 RGB、depth、fused 三者。

需要新增的代码对象有明确位置：

- depth P3：`tasks.py:252-255` 的 `_depth_save_map[4]`；
- RGB P3：`tasks.py:260-267` 的 `y[4]`；
- mask feature：新增 PMG 的中间特征或其投影；
- 对比损失：按 COFNet `ContrastLoss` 做 L2 normalize、相似度矩阵和双向 CE；
- 汇总位置：`loss.py:550-556` 主 loss 加权之后。

建议的损失定义只保留符号，不编造权重：

```text
L_new = L_S8 + λ_mask · L_mask + λ_dm · L_depth-mask
```

其中：

- `L_S8` 是第 3 节已证明的 box + seg + cls；
- `L_mask` 可先复现 COFNet 命令行默认的 BCE；其他形式必须单独消融；
- `L_depth-mask` 可复现 COFNet 的双向交叉熵；
- `λ_mask`、`λ_dm` 在 S8 上均未验证，不能直接沿用 COFNet 数值并写成最佳值。

### 5.6 建议 D：光照问题先做“可观测性”，再决定是否加门控

**[S8 已实现]** 原始 S8 是 `rgb + attention(rgb, depth)`，没有可解释的空间模态权重。仅凭漏检现象无法判断原因是 RGB 过强、depth 无效、配准误差还是标注/尺度问题。

**[待实现建议]** 在不改变预测的前提下，先记录以下张量在曝光/暗场/普通场景中的统计：

```text
||RGB_P3||, ||Depth_P3||,
||fused_P3 - RGB_P3|| / (||RGB_P3|| + eps),
PMG mask IoU/Recall,
按可见目标面积分桶的 Recall
```

若融合增量长期接近零，才能支持“depth 对融合贡献弱”；若 depth 特征本身异常，再回到第 2 节检查 dtype、归一化、padding 和配准。当前仓库没有这些统计结果，所以本文不宣称是哪一种原因。

---

## 6. 对“小目标、曝光、过暗、depth 不起作用”的严谨判断

| 用户观察            | 当前代码能证明什么                                                          | 当前代码不能证明什么                                    |
| ------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------- |
| 遮挡后只露一角漏检  | S8 最细检测层是 P3/stride 8；P3 有 wavelet edge gate，但没有 GT/object mask | 不能证明漏检一定由 P3 分辨率、loss 或融合模块造成       |
| 曝光叠件漏检        | 原始 S8 没有显式空间模态权重，RGB residual 固定保留                         | 不能证明换成 depth 就一定能检测到                       |
| 过暗黑包裹漏检      | depth 分支确实进入 P3/P4/P5 融合                                            | 不能证明 depth 数据在这些样本中有效、已配准或有正确尺度 |
| 感觉 depth 没起作用 | README 的 s3/s8 消融显示加入 Depth FPN 后整体指标有变化                     | 不能由整体 mAP 推断曝光/暗场子集的 depth 贡献           |

README 的可核验消融为 `README_RGBD_S8.md:306-320`：

| run | 结构                                | best Mask mAP50-95 | final Mask mAP50-95 |
| --- | ----------------------------------- | -----------------: | ------------------: |
| s3  | CoordAttV2 + P3 wavelet             |            0.79678 |             0.79603 |
| s8  | CoordAttV2 + P3 wavelet + Depth FPN |            0.80303 |             0.79863 |

按表中数值直接相减，s8 相对 s3 的 best 指标为 `+0.00625`，final 指标为 `+0.00260`。这只能说明 README 记录的整体指标存在小幅正差，不能替代难例子集实验。

---

## 7. 严格的实施与消融顺序

以下是实验计划，不是效果结论。

### E0：复现原始 S8

- 配置固定使用 `ultralytics/cfg/models/11/yolo11-seg.yaml`；
- 训练入口按 `train_genye.py:14-26`：`imgsz=640, epochs=300, batch=36, optimizer=MuSGD`；
- 确认所有新增开关关闭；
- 保存整体指标及固定难例列表预测。

### E1：只加 P3 PMG 与 mask supervision

- 不改变现有融合输出；
- 只验证 PMG 是否能从 P3 RGB-D 特征预测前景；
- 报告 `mask loss`、前景 IoU/Recall，按目标可见面积分桶。

### E2：P3 mask-guided fusion

- 在 E1 基础上让 mask 进入 P3 融合；
- 分别测试“按 COFNet wrapper 接口、但补上有效 mask 运算”的版本与一致路径变体；
- 其余配置与 E0 相同。

### E3：P3 depth-mask contrast

- 在 E2 基础上增加一项对比损失；
- 对比温度可先复现官方 `0.1`，但 loss 权重必须在 S8 上独立扫描；
- 不同时增加 P4/P5 或新检测头，保持单变量。

### E4：只在 E3 为正时扩展 P4

- P5 是否需要 mask 引导另做实验；
- 不把三尺度一次性改动的结果归因给某一个尺度。

### 必须增加的难例子集

整体 mAP 无法回答用户提出的三个具体问题。需要人工给验证图增加不参与训练的评估标签：

```text
occluded_corner / overexposed_stack / dark_black / normal
```

每组报告 Recall、漏检数、误检数和 mask IoU。当前仓库未找到这类场景标签，因此本文不提供虚构的场景指标。

---

## 8. 需要修改的文件与可核验落点

| 文件                                        | 现有证据位置          | 待实现内容                                                   |
| ------------------------------------------- | --------------------- | ------------------------------------------------------------ |
| `ultralytics/cfg/models/11/yolo11-seg.yaml` | `7-25`                | 新增明确且默认关闭的 PMG/contrast 开关                       |
| `ultralytics/nn/tasks.py`                   | `1925-2000`           | 定义 P3 PMG 或 mask-guided wrapper                           |
| `ultralytics/nn/tasks.py`                   | `260-267`             | 在 P3 融合调用点返回/使用 mask logits                        |
| `ultralytics/nn/tasks.py`                   | `2938-3001`           | 由 YAML 条件构建新模块                                       |
| `ultralytics/nn/tasks.py`                   | `445-458`             | 若严格复现 GT-guided train path，需要把 GT mask 送入 forward |
| `ultralytics/utils/loss.py`                 | `474-556`             | 汇总 `L_mask` 和可选 `L_depth-mask`                          |
| `ultralytics/utils/loss.py`                 | `385-409`             | 可复用实例 mask union/downsample 逻辑                        |
| `ultralytics/models/yolo/detect/train.py`   | `113-126`             | 仅在原始数据统计证明有问题后改 depth normalization           |
| `ultralytics/data/augment.py`               | `1388-1393,1806-1817` | 仅在 depth 无效值定义确认后改 padding/valid mask             |

所有新功能必须默认关闭，确保 E0 能完整复现 S8；不能直接覆盖 S8 配置后再把结果称为原始 S8。

---

## 9. 本次明确撤回的无依据表述

下列说法在现有 S8 证据下不成立，本文已删除：

1. “S8 depth 一定是 16-bit、范围约 1500–3400 mm。”——原始 `/workspace/Datasets/final` 不在当前工作区，无法核验。
2. “LetterBox 上下补边一定约占 25%。”——没有统计原始图像尺寸，无法核验。
3. “`114` 一定是异常近距离伪深度并导致漏检。”——不知道 depth 编码和无效值定义，只能列为审计点。
4. “spatial illumination gate 的 P3 RGB 权重约为某个值。”——它不是 README 定义的原始 S8。
5. “预训练只迁移 246/926，导致 depth 随机初始化。”——该日志不是严格的原始 S8 证据。
6. “S8 已启用 depth auxiliary loss/gate loss。”——YAML 没有总开关，代码默认关闭。
7. “COFNet 实际训练使用 `models/loss.py::ClipLoss`。”——官方训练入口实际导入 `utils/re_loss2.py::ContrastLoss`。
8. “加入 GT-mask/contrast 后一定提升小目标或暗场指标。”——尚无 S8 实验结果，不能作此结论。

---

## 10. 最终建议

在严格限定 S8 的前提下，最有依据的第一项改造不是直接堆叠完整 COFNet，而是：

```text
保持原始 S8 主干、Depth FPN、P3 wavelet 和检测头不变
→ 在 P3 增加独立 PMG
→ 用 S8 已有 GT instance mask 的 union 监督 PMG
→ 先证明 PMG 能识别前景
→ 再让预测/GT mask 引导 P3 融合
→ 最后单独加入 depth-mask contrast
```

这条顺序同时满足三点：有官方 COFNet 代码来源、有 S8 的明确接入位置、每一步都能通过单变量消融证伪。至于能否改善“只露一角、曝光叠件、暗色包裹”，必须由第 7 节难例子集实验回答，本文不提前编造结论。

---

## 11. 证据索引

### 本地 S8

- `D:\Graduate\project\genye\README_RGBD_S8.md`
- `D:\Graduate\project\genye\ultralytics\cfg\models\11\yolo11-seg.yaml`
- `D:\Graduate\project\genye\ultralytics\nn\tasks.py`
- `D:\Graduate\project\genye\ultralytics\data\build.py`
- `D:\Graduate\project\genye\ultralytics\data\dataset.py`
- `D:\Graduate\project\genye\ultralytics\data\augment.py`
- `D:\Graduate\project\genye\ultralytics\models\yolo\detect\train.py`
- `D:\Graduate\project\genye\ultralytics\models\yolo\detect\val.py`
- `D:\Graduate\project\genye\ultralytics\utils\loss.py`
- `D:\Graduate\project\genye\train_genye.py`

### COFNet 官方仓库

- [COFNet 官方仓库](https://github.com/li554/COFNet)
- [box-level mask 构造](https://github.com/li554/COFNet/blob/main/utils/general.py#L56-L72)
- [PMG 与 MaskGuideFusionBlock](https://github.com/li554/COFNet/blob/main/models/common.py#L2138-L2186)
- [NewCrossAttention 中被注释的 mask 运算](https://github.com/li554/COFNet/blob/main/models/conv_transformer.py#L295-L335)
- [模型池化输出](https://github.com/li554/COFNet/blob/main/models/yolo_mask_v2.py#L183-L212)
- [MaskLoss 与 ContrastLoss](https://github.com/li554/COFNet/blob/main/utils/re_loss2.py#L28-L72)
- [reconstruction loss 累加代码被注释](https://github.com/li554/COFNet/blob/main/utils/re_loss2.py#L417-L449)
- [真实训练 loss 组合](https://github.com/li554/COFNet/blob/main/train.py#L374-L389)
