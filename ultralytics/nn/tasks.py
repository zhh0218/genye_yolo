import contextlib
import math
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    E2ESegmentLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


def _init_yolov5_cmx_weights(m):
    """Initialization used by the old YOLOv5 FRM/FFM branch for stable transfer."""
    if isinstance(m, nn.Linear):
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(m.weight, std=0.02)
        else:
            nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class BaseModel(torch.nn.Module):
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, xd=None, *args, **kwargs):  # 行程1
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, xd, *args, **kwargs)
        # 到这里要停止
        return self.predict(x, xd, *args, **kwargs)

    def predict(self, x, xd=None, profile=False, visualize=False, augment=False, embed=None):  # 行程2
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, xd, profile, visualize, embed)

    def _predict_once(self, x, xd, profile=False, visualize=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            xd (torch.Tensor): The depth/thermal input tensor.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)

        _ds, _de = self._depth_range  # depth layer [start, end)
        if not hasattr(self, "_depth_fpn_index") or not hasattr(self, "_fusion_indices"):
            _extra_start = _de
            _use_depth_fpn = bool(getattr(self, "depth_fpn", False))
            self._depth_fpn_index = _extra_start if _use_depth_fpn else None
            _fusion_start = _extra_start + (1 if _use_depth_fpn else 0)
            self._fusion_indices = [_fusion_start + i for i in range(3)]
        if not hasattr(self, "rgbd_reliability_gate"):
            self.rgbd_reliability_gate = False
        if not hasattr(self, "_rgbd_reliability_gate_indices"):
            self._rgbd_reliability_gate_indices = []
        _depth_save_map = {4: None, 6: None, 10: None}
        yd = {}  # depth intermediate outputs keyed by depth-local stage index
        collect_rgbd_aux = self.training and self.model.training and torch.is_grad_enabled()
        if self.training:
            self._last_depth_aux_logits = None
            self._last_rgbd_gate_alphas = []

        # ── Phase 1: run both backbones, store depth P3/P4/P5 ──────────────
        # Run depth backbone independently first, storing all stage outputs
        for did in range(_ds, _de):
            md = self.model[did]
            if did == _ds:
                pass  # xd is the input depth tensor
            xd = md(xd)
            depth_local = did - _ds  # 0-based stage index within depth backbone
            yd[depth_local] = xd

        # Map: RGB stage → depth-local stage (they share the same backbone layout)
        for rgb_stage in _depth_save_map:
            _depth_save_map[rgb_stage] = yd[rgb_stage]

        # Run RGB backbone
        for m in self.model:
            if m.i > 10:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if m.i in self.save:
                y.append(x)
            else:
                y.append(None)

        # ── Phase 2: depth lightweight FPN (if enabled) ────────────────────
        if self.depth_fpn:
            depth_fpn_module = self.model[self._depth_fpn_index]
            p3d, p4d, p5d = depth_fpn_module(_depth_save_map[4], _depth_save_map[6], _depth_save_map[10])
            _depth_save_map[4], _depth_save_map[6], _depth_save_map[10] = p3d, p4d, p5d
        if collect_rgbd_aux and getattr(self, "_depth_aux_head_index", None) is not None:
            self._last_depth_aux_logits = self.model[self._depth_aux_head_index](_depth_save_map[4])

        # ── Phase 2.5: fuse at backbone outputs ───────────────────────────
        _fusion_keys = [4, 6, 10]
        for fi, key in enumerate(_fusion_keys):
            rgb_feat, depth_feat = y[key], _depth_save_map[key]
            if self.rgbd_reliability_gate:
                gate_module = self.model[self._rgbd_reliability_gate_indices[fi]]
                rgb_feat, depth_feat = gate_module(rgb_feat, depth_feat)
            fusion_module = self.model[self._fusion_indices[fi]]
            y[key] = fusion_module(rgb_feat, depth_feat)
            if collect_rgbd_aux and getattr(self, "rgbd_gate_loss", False):
                alpha = self._last_fusion_alpha(fusion_module)
                if alpha is not None:
                    self._last_rgbd_gate_alphas.append((key, alpha))

        # ── Phase 3: RGB FPN head (layers 11-23) ──────────────────────────
        x = y[10]  # resume from last backbone output (P5, already in y)
        for m in self.model:
            if m.i <= 10:
                continue
            if m.i > 23:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

            if m.i in self.save:
                y.append(x)
            else:
                y.append(None)

        return x

    @staticmethod
    def _last_fusion_alpha(module):
        """Return the latest differentiable modality alpha stored by a fusion module."""
        for child in module.modules():
            alpha = getattr(child, "_last_alpha_train", None)
            if alpha is not None:
                return alpha
        return None

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[23]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, depth_batch=None, preds=None):  # 行程1.1
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        # 创建损失函数并传入img数据，然后再次回到forward函数
        depth = (
            depth_batch["img"]
            if isinstance(depth_batch, dict)
            else batch.get("depth_img")
            if depth_batch is None
            else depth_batch
        )
        preds = self.forward(batch["img"], depth) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[23], "end2end", False)
        self.prog_loss = self.yaml.get("prog_loss", False)
        self.stal = self.yaml.get("stal", False)
        self.depth_fpn = self.yaml.get("depth_fpn", False)
        self.rgbd_reliability_gate = bool(self.yaml.get("rgbd_reliability_gate", False))
        self.rgbd_reliability_mode = self.yaml.get("rgbd_reliability_mode", "relative_soft")
        self.rgbd_aux_loss = bool(self.yaml.get("rgbd_aux_loss", False))
        self.rgbd_gate_loss = self.rgbd_aux_loss and bool(self.yaml.get("rgbd_gate_loss", True))
        self.rgbd_depth_aux_loss = self.rgbd_aux_loss and bool(self.yaml.get("rgbd_depth_aux_loss", True))

        # Depth backbone layer mapping — derived from YAML structure, not hardcoded
        _n_backbone = len(self.yaml.get("backbone", []))
        _n_head = len(self.yaml.get("head", []))
        _depth_offset = _n_backbone + _n_head  # first depth layer index in self.model
        _n_depth = len(self.yaml.get("backboneD", []))
        self._depth_range = (_depth_offset, _depth_offset + _n_depth)  # [start, end)
        _rgbd_extra_start = _depth_offset + _n_depth
        self._depth_fpn_index = _rgbd_extra_start if self.depth_fpn else None
        _fusion_start = _rgbd_extra_start + (1 if self.depth_fpn else 0)
        self._fusion_indices = [_fusion_start + i for i in range(3)]
        _gate_start = _fusion_start + 3
        self._rgbd_reliability_gate_indices = [_gate_start + i for i in range(3)] if self.rgbd_reliability_gate else []
        _aux_start = _gate_start + (3 if self.rgbd_reliability_gate else 0)
        self._depth_aux_head_index = _aux_start if self.rgbd_depth_aux_loss else None
        # RGB backbone stage → absolute depth layer index (P3=4, P4=6, P5=10)
        self._depth_stage_idx = {stage: _depth_offset + stage for stage in (4, 6, 10)}

        # Build strides
        m = self.model[23]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x, xd):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    out = self.forward(x, xd)
                    one2many = out["one2many"]
                    # For Segment end2end, one2many is (feats, mc, p); for Detect end2end, it's feats list
                    return one2many[0] if isinstance(one2many, tuple) else one2many
                return (
                    self.forward(x, xd)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x, xd)
                )

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))]
            )  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        stal = getattr(self, "stal", False)
        if getattr(self, "end2end", False):
            return E2EDetectLoss(self, prog_loss=getattr(self, "prog_loss", False), stal=stal)
        return v8DetectionLoss(self, stal=stal)


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        stal = getattr(self, "stal", False)
        if getattr(self, "end2end", False):
            return E2ESegmentLoss(self, prog_loss=getattr(self, "prog_loss", False), stal=stal)
        return v8SegmentationLoss(self, stal=stal)


class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (list[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (list[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load a single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


# 融合模块开始
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup // 2, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)

        _n, _c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)

        return out


# 融合模块结束


# ─────────────────────────────────────────────────────────────────────────────
# CoordAtt V2  — fixes residual / SA redundancy / channel reduction issues
# ─────────────────────────────────────────────────────────────────────────────


class CoordAttV2(nn.Module):
    """Improved coordinate attention for RGB-D fusion. vs V1: residual connection, no redundant SA_Enhance, proper
    channel projection.

    When adaptive_gate=True, the hard-coded `rgb + out` residual is replaced by a learned modality gate: α·rgb +
    (1−α)·depth + out. gate_mode controls how α is predicted: 'channel' — global per-channel, α is [B, half, 1, 1]
    (original). 'spatial' — region-level single map, α is [B, 1, H, W]. 'spatial_channel' — region-level per-channel, α
    is [B, half, H, W]. 'spatial_illum' — 'spatial' + local brightness/contrast prior of rgb.
    """

    def __init__(
        self,
        inp,
        oup,
        reduction=32,
        adaptive_gate=False,
        gate_mode="channel",
        base_scale=1.0,
        out_scale=1.0,
        learnable_blend=False,
        blend_init=0.5,
    ):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Sequential(
            nn.Conv2d(inp, oup // 2, 1, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.SiLU(inplace=True),
        )

        self.adaptive_gate = adaptive_gate
        self.gate_mode = gate_mode
        self.base_scale = float(base_scale)
        self.out_scale = float(out_scale)
        self.learnable_blend = bool(learnable_blend)
        if self.learnable_blend:
            blend_init = float(blend_init)
            blend_init = min(max(blend_init, 1e-4), 1.0 - 1e-4)
            self.blend_logits = nn.Parameter(torch.log(torch.tensor([blend_init, 1.0 - blend_init])))
        if adaptive_gate:
            half = inp // 2  # channels per modality
            mid = max(8, half // 4)
            if gate_mode == "channel":
                # global per-channel gate (original): alpha is [B, half, 1, 1]
                self.mod_gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(inp, mid, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(mid, half, bias=False),
                    nn.Sigmoid(),
                )
            elif gate_mode in ("spatial", "spatial_channel", "spatial_illum"):
                # region-level reliability gate: alpha keeps spatial dims [B, *, H, W].
                # lightweight head: 1x1 reduce -> depthwise 3x3 (spatial context) -> 1x1 expand.
                in_ch = inp + (2 if gate_mode == "spatial_illum" else 0)
                out_ch = 1 if gate_mode in ("spatial", "spatial_illum") else half
                mid_s = max(8, inp // 16)
                self.mod_gate = nn.Sequential(
                    nn.Conv2d(in_ch, mid_s, 1, bias=False),
                    nn.BatchNorm2d(mid_s),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_s, mid_s, 3, padding=1, groups=mid_s, bias=False),
                    nn.BatchNorm2d(mid_s),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_s, out_ch, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                raise ValueError(f"unknown gate_mode={gate_mode}")

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)

        _n, _c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = self.proj(x * a_w * a_h)

        self._last_alpha_train = None
        if self.adaptive_gate:
            gm = getattr(self, "gate_mode", "channel")  # back-compat: old ckpts lack gate_mode
            if gm == "channel":
                alpha = self.mod_gate(x).unsqueeze(-1).unsqueeze(-1)  # [B, half, 1, 1]
            else:
                g_in = x
                if gm == "spatial_illum":
                    g_in = torch.cat([x, self._illum_prior(rgb.detach())], dim=1)
                alpha = self.mod_gate(g_in)  # [B, 1 or half, H, W] region-level map
            self._last_alpha = alpha.detach()  # for visualization/analysis; not in graph
            self._last_alpha_train = alpha if self.training and torch.is_grad_enabled() else None
            base = alpha * rgb + (1 - alpha) * depth
            if self.learnable_blend:
                blend = torch.softmax(self.blend_logits, dim=0)
                return blend[0] * base + blend[1] * out
            return self.base_scale * base + self.out_scale * out
        if self.learnable_blend:
            blend = torch.softmax(self.blend_logits, dim=0)
            return blend[0] * rgb + blend[1] * out
        return self.base_scale * rgb + self.out_scale * out

    @staticmethod
    def _illum_prior(rgb):
        # cheap RGB reliability prior: local brightness + local contrast (dark/over-exposed/low-texture -> RGB less reliable)
        b = rgb.mean(dim=1, keepdim=True)
        mu = F.avg_pool2d(b, 7, 1, 3)
        var = (F.avg_pool2d(b * b, 7, 1, 3) - mu * mu).clamp_min(0.0)
        return torch.cat([mu, torch.sqrt(var + 1e-6)], dim=1)


class RGBDCoordAttV2(nn.Module):
    """CoordAttV2 adapter for the unified rgbd_fusion API."""

    def __init__(
        self,
        c_in,
        num_heads=8,
        kv_pool=10,
        adaptive_gate=False,
        gate_mode="channel",
        base_scale=1.0,
        out_scale=1.0,
        learnable_blend=False,
        blend_init=0.5,
    ):
        super().__init__()
        self.coord_att = CoordAttV2(
            c_in * 2,
            c_in * 2,
            adaptive_gate=adaptive_gate,
            gate_mode=gate_mode,
            base_scale=base_scale,
            out_scale=out_scale,
            learnable_blend=learnable_blend,
            blend_init=blend_init,
        )

    def forward(self, rgb, depth):
        return self.coord_att(rgb, depth)


class RGBDWaveletGuidedCoordAttV2(nn.Module):
    """Wavelet-guided fusion using CoordAttV2 as the base cross-modal block."""

    def __init__(
        self,
        c_in,
        num_heads=8,
        kv_pool=10,
        adaptive_gate=False,
        gate_mode="channel",
        base_scale=1.0,
        out_scale=1.0,
        learnable_blend=False,
        blend_init=0.5,
    ):
        super().__init__()
        self.low_proj = Conv(c_in, c_in, 1, 1)
        self.hf_reduce = Conv(c_in * 3, c_in, 1, 1)
        self.hf_refine = Conv(c_in, c_in, 3, 1)
        self.edge_gate = nn.Conv2d(c_in, 1, kernel_size=1, stride=1, padding=0)
        self.coord_att = CoordAttV2(
            c_in * 2,
            c_in * 2,
            adaptive_gate=adaptive_gate,
            gate_mode=gate_mode,
            base_scale=base_scale,
            out_scale=out_scale,
            learnable_blend=learnable_blend,
            blend_init=blend_init,
        )
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

    def _haar_split(self, x):
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (-x00 - x01 + x10 + x11) * 0.25
        hl = (-x00 + x01 - x10 + x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def forward(self, rgb, depth):
        target_size = rgb.shape[-2:]
        ll, lh, hl, hh = self._haar_split(depth)
        depth_low = torch.nn.functional.interpolate(self.low_proj(ll), size=target_size, mode="nearest")
        edge_feat = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
        edge_gate = torch.sigmoid(
            torch.nn.functional.interpolate(self.edge_gate(edge_feat), size=target_size, mode="nearest")
        )
        base = self.coord_att(rgb, depth_low)
        return base + self.beta * base * edge_gate


# ─────────────────────────────────────────────────────────────────────────────
# Depth Lightweight FPN  (enabled via  depth_fpn: True  in YAML)
# ─────────────────────────────────────────────────────────────────────────────


class DepthLightFPN(nn.Module):
    """Top-down FPN for the depth branch — gives depth multi-scale context before fusion."""

    def __init__(self, ch_p3, ch_p4, ch_p5):
        super().__init__()
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce5to4 = Conv(ch_p5 + ch_p4, ch_p4, 1, 1)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce4to3 = Conv(ch_p4 + ch_p3, ch_p3, 1, 1)

    def forward(self, p3, p4, p5):
        p4 = self.reduce5to4(torch.cat([self.up5(p5), p4], dim=1))
        p3 = self.reduce4to3(torch.cat([self.up4(p4), p3], dim=1))
        return p3, p4, p5


class RGBDDepthAuxHead(nn.Module):
    """Lightweight train-only foreground head on the depth P3 feature."""

    def __init__(self, c_in, hidden=None):
        super().__init__()
        hidden = hidden or max(c_in // 2, 16)
        self.head = nn.Sequential(
            Conv(c_in, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.head(x)


class RGBDRelativeReliabilityGate(nn.Module):
    """Identity-initialized soft gate for relative RGB/depth feature reliability."""

    def __init__(self, c_in, mode="relative_soft", reduction=16, max_delta=0.5):
        super().__init__()
        if mode != "relative_soft":
            raise ValueError(f"Unsupported rgbd_reliability_mode: {mode}")
        hidden = max(c_in // reduction, 16)
        self.mode = mode
        self.max_delta = float(max_delta)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c_in * 4, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, rgb, depth):
        context = torch.cat((rgb, depth, torch.abs(rgb - depth), rgb * depth), dim=1)
        logits = self.mlp(self.pool(context))
        scales = 1.0 + self.max_delta * torch.tanh(logits)
        rgb_scale, depth_scale = scales[:, 0:1], scales[:, 1:2]
        return rgb * rgb_scale, depth * depth_scale


# ─────────────────────────────────────────────────────────────────────────────
# RGB-D Fusion modules  (selectable via  rgbd_fusion  in yolo11-seg.yaml)
# Options: 'se' | 'mamba' | 'cross_v1' | 'cross_v2' | 'coord_att' | 'cmm'
# ─────────────────────────────────────────────────────────────────────────────


class RGBDCoordAtt(nn.Module):
    """CoordAtt adapter for the unified rgbd_fusion API.

    Wraps CoordAtt(inp=c_in*2, oup=c_in*2) so it accepts the standard (c_in, num_heads, kv_pool) constructor and
    forward(rgb, depth) signature. Internally: cat(rgb, depth) → CoordAtt coordinate attention → 512ch output.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        self.coord_att = CoordAtt(c_in * 2, c_in * 2)

    def forward(self, rgb, depth):
        return self.coord_att(rgb, depth)


class RGBDConcatFusion(nn.Module):
    """Simple concat baseline: cat(RGB, X) followed by a 1x1 projection."""

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        self.proj = Conv(c_in * 2, c_in, 1, 1)

    def forward(self, rgb, depth):
        return self.proj(torch.cat((rgb, depth), dim=1))


class RGBDAddFusion(nn.Module):
    """Simple add baseline; requires aligned RGB/X feature channels."""

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()

    def forward(self, rgb, depth):
        return rgb + depth


class _YOLOv5ChannelWeights(nn.Module):
    """Channel reweighting from the old YOLOv5 RGB-D fusion branch."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        hidden = max(dim * 4 // reduction, 8)
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        b = x1.shape[0]
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(b, self.dim * 2)
        max_ = self.max_pool(x).view(b, self.dim * 2)
        y = self.mlp(torch.cat((avg, max_), dim=1)).view(b, self.dim * 2, 1)
        return y.reshape(b, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class _YOLOv5SpatialWeights(nn.Module):
    """Spatial reweighting from the old YOLOv5 RGB-D fusion branch."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        b, _, h, w = x1.shape
        spatial_weights = self.mlp(torch.cat((x1, x2), dim=1))
        return spatial_weights.reshape(b, 2, 1, h, w).permute(1, 0, 2, 3, 4)


class _YOLOv5FeatureRectifyModule(nn.Module):
    """Old YOLOv5 FRM: mutually rectifies RGB and auxiliary features."""

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = _YOLOv5ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = _YOLOv5SpatialWeights(dim=dim, reduction=reduction)
        self.apply(_init_yolov5_cmx_weights)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return torch.nan_to_num(out_x1), torch.nan_to_num(out_x2)


class _YOLOv5CrossAttention(nn.Module):
    """Linear cross-attention used by the old YOLOv5 FeatureFusionModule."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        num_heads = max(1, min(num_heads, dim))
        while dim % num_heads != 0:
            num_heads -= 1
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        dtype = x1.dtype
        b, n, c = x1.shape
        h = self.num_heads
        d = c // h
        q1 = x1.reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(b, n, 2, h, d).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(b, n, 2, h, d).permute(2, 0, 3, 1, 4).contiguous()

        # The old FFM attention is numerically fragile in AMP; do matmul/softmax in FP32.
        q1, q2 = q1.float(), q2.float()
        k1, v1, k2, v2 = k1.float(), v1.float(), k2.float(), v2.float()
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx1 = torch.nan_to_num(ctx1, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-50, 50)
        ctx2 = torch.nan_to_num(ctx2, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-50, 50)
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = ctx2.softmax(dim=-2)
        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        return torch.nan_to_num(x1).to(dtype), torch.nan_to_num(x2).to(dtype)


class _YOLOv5CrossPath(nn.Module):
    """Old YOLOv5 cross-path fusion before channel embedding."""

    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.channel_proj1 = nn.Linear(dim, hidden * 2)
        self.channel_proj2 = nn.Linear(dim, hidden * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = _YOLOv5CrossAttention(hidden, num_heads=num_heads)
        self.end_proj1 = nn.Linear(hidden * 2, dim)
        self.end_proj2 = nn.Linear(hidden * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        out_x1 = self.norm1(x1 + self.end_proj1(torch.cat((y1, v1), dim=-1)))
        out_x2 = self.norm2(x2 + self.end_proj2(torch.cat((y2, v2), dim=-1)))
        return torch.nan_to_num(out_x1), torch.nan_to_num(out_x2)


class _YOLOv5ChannelEmbed(nn.Module):
    """Converts fused token features back to a dense feature map."""

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        hidden = max(out_channels // reduction, 8)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        return torch.nan_to_num(self.norm(self.residual(x) + self.channel_embed(x)))


class _YOLOv5FeatureFusionModule(nn.Module):
    """Old YOLOv5 FFM: cross-path attention followed by channel embedding."""

    def __init__(self, dim, reduction=1, num_heads=8):
        super().__init__()
        self.cross = _YOLOv5CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = _YOLOv5ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction)
        self.apply(_init_yolov5_cmx_weights)

    def forward(self, x1, x2):
        _b, _c, h, w = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        return torch.nan_to_num(self.channel_emb(torch.cat((x1, x2), dim=-1), h, w))


class RGBDYOLOv5FRMFFMFusion(nn.Module):
    """Port of the old YOLOv5 FRM + FFM RGB-D fusion block."""

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        self.frm = _YOLOv5FeatureRectifyModule(dim=c_in)
        self.ffm = _YOLOv5FeatureFusionModule(dim=c_in, num_heads=num_heads)

    def forward(self, rgb, depth):
        rgb, depth = self.frm(rgb, depth)
        return torch.nan_to_num(self.ffm(rgb, depth))


class RGBDWaveletGuidedCoordAtt(nn.Module):
    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        self.low_proj = Conv(c_in, c_in, 1, 1)
        self.hf_reduce = Conv(c_in * 3, c_in, 1, 1)
        self.hf_refine = Conv(c_in, c_in, 3, 1)
        self.edge_gate = nn.Conv2d(c_in, 1, kernel_size=1, stride=1, padding=0)
        self.coord_att = CoordAtt(c_in * 2, c_in * 2)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

    def _haar_split(self, x):
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (-x00 - x01 + x10 + x11) * 0.25
        hl = (-x00 + x01 - x10 + x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def forward(self, rgb, depth):
        target_size = rgb.shape[-2:]
        ll, lh, hl, hh = self._haar_split(depth)
        depth_low = torch.nn.functional.interpolate(self.low_proj(ll), size=target_size, mode="nearest")
        edge_feat = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
        edge_gate = torch.sigmoid(
            torch.nn.functional.interpolate(self.edge_gate(edge_feat), size=target_size, mode="nearest")
        )
        base = self.coord_att(rgb, depth_low)
        return base + self.beta * base * edge_gate


class RGBDWaveletHFFusion(nn.Module):
    def __init__(self, c_in, num_heads=8, kv_pool=10, base_fusion_cls=RGBDCoordAtt, **fusion_kwargs):
        super().__init__()
        self.base_fusion = base_fusion_cls(c_in, num_heads=num_heads, kv_pool=kv_pool, **fusion_kwargs)
        self.hf_reduce = Conv(c_in * 3, c_in, 1, 1)
        self.hf_refine = Conv(c_in, c_in, 3, 1)

    def _haar_split(self, x):
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (-x00 - x01 + x10 + x11) * 0.25
        hl = (-x00 + x01 - x10 + x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def forward(self, rgb, depth):
        target_size = rgb.shape[-2:]
        _, lh, hl, hh = self._haar_split(depth)
        depth_high = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
        depth_high = torch.nn.functional.interpolate(depth_high, size=target_size, mode="nearest")
        return self.base_fusion(rgb, depth_high)


class RGBDWaveletAdaptiveFusion(nn.Module):
    """Adaptive wavelet fusion — uses BOTH low-freq and high-freq bands from Haar decomposition, with a learned gate to
    balance their contribution. Replaces RGBDWaveletHFFusion which discards LL entirely.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, base_fusion_cls=RGBDCoordAtt, **fusion_kwargs):
        super().__init__()
        self.base_fusion = base_fusion_cls(c_in, num_heads=num_heads, kv_pool=kv_pool, **fusion_kwargs)
        self.lf_proj = Conv(c_in, c_in, 1, 1)
        self.hf_reduce = Conv(c_in * 3, c_in, 1, 1)
        self.hf_refine = Conv(c_in, c_in, 3, 1)
        # Learned gate: squeeze both branches → per-channel blend weight
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in * 2, c_in // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in // 4, c_in, 1, bias=False),
            nn.Sigmoid(),
        )

    def _haar_split(self, x):
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (-x00 - x01 + x10 + x11) * 0.25
        hl = (-x00 + x01 - x10 + x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def forward(self, rgb, depth):
        target_size = rgb.shape[-2:]
        ll, lh, hl, hh = self._haar_split(depth)
        # Low-freq: structural / semantic content
        depth_low = self.lf_proj(ll)
        depth_low = torch.nn.functional.interpolate(depth_low, size=target_size, mode="nearest")
        # High-freq: edges / texture detail
        depth_high = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
        depth_high = torch.nn.functional.interpolate(depth_high, size=target_size, mode="nearest")
        # Adaptive blend: gate ∈ (0,1), gate→1 favors low-freq, gate→0 favors high-freq
        alpha = self.gate(torch.cat([depth_low, depth_high], dim=1))
        depth_mixed = alpha * depth_low + (1 - alpha) * depth_high
        return self.base_fusion(rgb, depth_mixed)


class RGBDScaleAwareWaveletFusion(nn.Module):
    def __init__(
        self,
        c_in,
        num_heads=8,
        kv_pool=10,
        base_fusion_cls=RGBDCoordAtt,
        band="raw",
        use_gate=False,
        low_gain=1.0,
        high_gain=1.0,
        **fusion_kwargs,
    ):
        super().__init__()
        band_alias = {
            "none": "raw",
            "raw": "raw",
            "rgbd": "raw",
            "low": "lf",
            "lf": "lf",
            "high": "hf",
            "hf": "hf",
            "both": "lfhf",
            "lfhf": "lfhf",
        }
        self.band = band_alias.get(str(band).lower(), "raw")
        self.low_gain = low_gain
        self.high_gain = high_gain
        self.base_fusion = base_fusion_cls(c_in, num_heads=num_heads, kv_pool=kv_pool, **fusion_kwargs)
        self.use_low = self.band in {"lf", "lfhf"}
        self.use_high = self.band in {"hf", "lfhf"}
        self.low_proj = Conv(c_in, c_in, 1, 1) if self.use_low else None
        self.hf_reduce = Conv(c_in * 3, c_in, 1, 1) if self.use_high else None
        self.hf_refine = Conv(c_in, c_in, 3, 1) if self.use_high else None
        self.mix_proj = Conv(c_in * 2, c_in, 1, 1) if self.band == "lfhf" else None
        if use_gate:
            gate_hidden = max(c_in // 4, 8)
            self.gate = nn.Sequential(
                nn.Conv2d(c_in * 2, gate_hidden, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, 1, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.gate = None

    def _haar_split(self, x):
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = torch.nn.functional.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (-x00 - x01 + x10 + x11) * 0.25
        hl = (-x00 + x01 - x10 + x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def _band_depth(self, depth, target_size):
        if self.band == "raw":
            if depth.shape[-2:] == target_size:
                return depth
            return torch.nn.functional.interpolate(depth, size=target_size, mode="nearest")
        ll, lh, hl, hh = self._haar_split(depth)
        parts = []
        if self.use_low:
            depth_low = torch.nn.functional.interpolate(self.low_proj(ll), size=target_size, mode="nearest")
            parts.append(depth_low * self.low_gain)
        if self.use_high:
            depth_high = self.hf_refine(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
            depth_high = torch.nn.functional.interpolate(depth_high, size=target_size, mode="nearest")
            parts.append(depth_high * self.high_gain)
        if len(parts) == 1:
            return parts[0]
        return self.mix_proj(torch.cat(parts, dim=1))

    def forward(self, rgb, depth):
        depth_band = self._band_depth(depth, rgb.shape[-2:])
        fused = self.base_fusion(rgb, depth_band)
        if self.gate is None:
            return fused
        gate = torch.sigmoid(self.gate(torch.cat((rgb, depth_band), dim=1)))
        return rgb + (fused - rgb) * gate


class RGBDCrossAttentionV1(nn.Module):
    """Full-dimension RGB-D cross-attention (v1 original). Q=RGB, KV=depth (pooled to kv_pool×kv_pool). Channel SE gate
    + spatial attn. Highest capacity — most prone to overfitting on small datasets.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        assert c_in % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = c_in // num_heads
        self.scale = self.head_dim**-0.5
        self.kv_pool = kv_pool

        self.norm_rgb = nn.GroupNorm(min(32, c_in // 16), c_in)
        self.norm_depth = nn.GroupNorm(min(32, c_in // 16), c_in)
        self.proj_q = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.proj_k = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.proj_v = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.proj_out = nn.Conv2d(c_in, c_in, 1, bias=False)

        r = max(c_in // 16, 8)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, c_in, 1, bias=False),
            nn.Tanh(),
        )
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        h, d = self.num_heads, self.head_dim

        rgb_ch = rgb + rgb * self.ch_gate(depth)
        depth_n = self.norm_depth(depth)
        Q = self.proj_q(self.norm_rgb(rgb))
        K = self.proj_k(depth_n)
        V = self.proj_v(depth_n)

        if H > self.kv_pool or W > self.kv_pool:
            K = F.adaptive_avg_pool2d(K, self.kv_pool)
            V = F.adaptive_avg_pool2d(V, self.kv_pool)
        Kh, Kw = K.shape[-2:]

        Q = Q.reshape(B, h, d, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, h, d, Kh * Kw).permute(0, 1, 3, 2)
        V = V.reshape(B, h, d, Kh * Kw).permute(0, 1, 3, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ V).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return rgb_ch + self.gamma * self.proj_out(out)


class RGBDCrossAttentionV2(nn.Module):
    """Reduced-dim RGB-D cross-attention (v2). QKV projected to c_in//2 (half params vs V1), attention dropout=0.1.
    Better regularized than V1 but still more params than SE/Mamba.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        c_mid = c_in // 2
        assert c_mid % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = c_mid // num_heads
        self.scale = self.head_dim**-0.5
        self.kv_pool = kv_pool
        self.c_mid = c_mid

        self.norm_rgb = nn.GroupNorm(min(32, c_in // 16), c_in)
        self.norm_depth = nn.GroupNorm(min(32, c_in // 16), c_in)
        self.proj_q = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.proj_k = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.proj_v = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, c_in, 1, bias=False)
        self.attn_drop = nn.Dropout(0.1)

        r = max(c_in // 16, 8)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, c_in, 1, bias=False),
            nn.Tanh(),
        )
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, rgb, depth):
        B, _C, H, W = rgb.shape
        h, d = self.num_heads, self.head_dim

        rgb_ch = rgb + rgb * self.ch_gate(depth)
        depth_n = self.norm_depth(depth)
        Q = self.proj_q(self.norm_rgb(rgb))
        K = self.proj_k(depth_n)
        V = self.proj_v(depth_n)

        if H > self.kv_pool or W > self.kv_pool:
            K = F.adaptive_avg_pool2d(K, self.kv_pool)
            V = F.adaptive_avg_pool2d(V, self.kv_pool)
        Kh, Kw = K.shape[-2:]

        Q = Q.reshape(B, h, d, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, h, d, Kh * Kw).permute(0, 1, 3, 2)
        V = V.reshape(B, h, d, Kh * Kw).permute(0, 1, 3, 2)

        attn = self.attn_drop((Q @ K.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ V).permute(0, 1, 3, 2).reshape(B, self.c_mid, H, W)
        return rgb_ch + self.gamma * self.proj_out(out)


class RGBDMambaFusion(nn.Module):
    """Mamba-inspired cross-modal fusion for RGB-D (pure PyTorch, no mamba-ssm).

    Depth features are scanned row-wise and col-wise as 1D sequences via depthwise Conv1d, approximating SSM sequential
    state propagation. Produces a spatially-aware (H×W) gate — unlike SE (single vector per channel) this preserves
    spatial structure. O(N) complexity vs O(N²) for cross-attention.

    Args:
        c_in (int): Channel dimension of both RGB and Depth features.
        num_heads (int): Unused, kept for API compatibility.
        kv_pool (int): Unused, kept for API compatibility.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        d = max(c_in // 4, 32)  # inner scan channels
        self.proj_in = nn.Conv2d(c_in, d, 1, bias=False)
        self.row_scan = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d, bias=False)
        self.col_scan = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d, bias=False)
        self.mix = nn.Conv2d(d * 2, d, 1, bias=False)
        self.proj_gate = nn.Sequential(
            nn.Conv2d(d, c_in, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, rgb, depth):
        B, _C, H, W = depth.shape
        x = self.proj_in(depth)  # (B, d, H, W)
        d = x.shape[1]

        # Row scan: each row is a length-W sequence
        x_row = x.view(B * H, d, W)
        x_row = self.row_scan(x_row).view(B, d, H, W)

        # Col scan: each col is a length-H sequence
        x_col = x.permute(0, 1, 3, 2).contiguous().view(B * W, d, H)
        x_col = self.col_scan(x_col).view(B, d, W, H).permute(0, 1, 3, 2)

        gate = self.proj_gate(self.mix(torch.cat([x_row, x_col], dim=1)))
        return rgb + rgb * gate  # spatial-aware depth gating, identity when gate=0


class RGBDCrossAttention(nn.Module):
    """Depth-guided Channel Gate for RGB-D feature fusion (cross-modal SE, default).

    Depth features are globally squeezed and used to generate a per-channel gate that modulates RGB features. Tanh gate
    ∈ (-1, 1) allows the depth context to both enhance and suppress individual RGB channels. Output = rgb + rgb *
    tanh(gate(depth)), which is identity when gate=0.

    Args:
        c_in (int): Channel dimension of both RGB and Depth features.
        num_heads (int): Unused, kept for API compatibility.
        kv_pool (int): Unused, kept for API compatibility.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        r = max(c_in // 16, 8)  # SE reduction ratio
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1) — global depth context
            nn.Conv2d(c_in, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, c_in, 1, bias=False),
            nn.Tanh(),  # gate ∈ (-1, 1): can enhance or suppress
        )

    def forward(self, rgb, depth):
        gate = self.ch_gate(depth)  # (B, C, 1, 1)
        return rgb + rgb * gate  # rgb * (1 + gate); identity when gate=0


class _CMMBlock(nn.Module):
    """Lightweight Mamba-like block for 2-D feature maps (pure PyTorch).

    Flattens H×W into a 1-D spatial sequence, applies: LayerNorm → Linear(expand) → DepthwiseConv1d + SiLU →
    Linear(contract) then adds a residual connection and reshapes back to (B,C,H,W). Approximates SSM sequential state
    propagation without mamba-ssm.
    """

    def __init__(self, c_in, expand=2, kernel_size=7):
        super().__init__()
        c_mid = c_in * expand
        self.norm = nn.LayerNorm(c_in)
        self.in_proj = nn.Linear(c_in, c_mid, bias=False)
        self.dw_conv = nn.Conv1d(c_mid, c_mid, kernel_size, padding=kernel_size // 2, groups=c_mid, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(c_mid, c_in, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_proj = self.in_proj(x_norm)  # (B, N, c_mid)
        x_conv = self.dw_conv(x_proj.transpose(1, 2))  # (B, c_mid, N)
        x_act = self.act(x_conv).transpose(1, 2)  # (B, N, c_mid)
        x_out = self.out_proj(x_act)  # (B, N, C)
        return (x_flat + x_out).reshape(B, H, W, C).permute(0, 3, 1, 2)


class RGBDCrossModalMamba(nn.Module):
    """CMM (Cross-Modal Fusion Mamba) from MambaSOD (arXiv 2410.15015).

    Three-branch design (all branches use _CMMBlock as a lightweight Mamba):
    1. Self-RGB   : RGB   → _CMMBlock(C)   → y_r
    2. Self-Depth : Depth → _CMMBlock(C)   → y_d
    3. Joint      : cat(RGB, Depth) → _CMMBlock(2C) → gate_proj → g  (C ch)

    Gated fusion: rgb + γ × (y_r × σ(g) + y_d × σ(g)) γ is a learnable scalar initialized to 0 (identity at init →
    stable training).

    Args:
        c_in (int): Channel dimension of both RGB and Depth features.
        num_heads (int): Unused, kept for API compatibility.
        kv_pool (int): Unused, kept for API compatibility.
    """

    def __init__(self, c_in, num_heads=8, kv_pool=10, **kwargs):
        super().__init__()
        # self.block_r     = _CMMBlock(c_in)
        # self.block_d     = _CMMBlock(c_in)
        # self.block_joint = _CMMBlock(c_in * 2)
        self.block_r = _CMMBlock(c_in, expand=1)
        self.block_d = _CMMBlock(c_in, expand=1)
        self.block_joint = _CMMBlock(c_in * 2, expand=1)
        self.gate_proj = nn.Conv2d(c_in * 2, c_in, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, depth):
        y_r = self.block_r(rgb)  # (B, C, H, W)
        y_d = self.block_d(depth)  # (B, C, H, W)
        g = self.gate_proj(self.block_joint(torch.cat([rgb, depth], dim=1)))  # (B, C, H, W)
        gate = torch.sigmoid(g)
        return rgb + self.gamma * (y_r * gate + y_d * gate)


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    end2end = d.get("end2end", False)
    reg_max = d.get("reg_max", 16)
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"] + d["backboneD"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if i == 24:  # backboneD开始的输入通道应该是1;所以在这里更改
                c1 = 3
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment}:
                # Insert reg_max and end2end before the ch argument
                # Detect(nc, reg_max, end2end, ch) / Segment(nc, nm, npr, reg_max, end2end, ch)
                if m is Detect:
                    args.insert(1, reg_max)  # args = [nc, reg_max, ch] -> insert end2end
                    args.insert(2, end2end)  # args = [nc, reg_max, end2end, ch]
                elif m is Segment:
                    args.insert(3, reg_max)  # args = [nc, nm, npr, reg_max, ch]
                    args.insert(4, end2end)  # args = [nc, nm, npr, reg_max, end2end, ch]
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # ── RGB-D Fusion: selectable via 'rgbd_fusion' in yolo11-seg.yaml ──────────
    # 'se'        — channel SE gate (default, ~98K params total, fastest)
    # 'mamba'     — row+col spatial scan gate (~496K, spatial-aware)
    # 'cross_v1'  — full cross-attention (~3.24M, highest capacity)
    # 'cross_v2'  — reduced cross-attention + dropout (~1.68M, regularized)
    # 'coord_att' — CoordAtt coordinate attention (~0.62M, best empirical)
    # 'concat'    — simple concat + 1x1 projection baseline
    # 'add'       — simple element-wise addition baseline
    # 'old_cmx_ffm' — old YOLOv5 FRM + FFM fusion port
    # 'cmm'       — MambaSOD CMM: self-enhance + joint gate (~1.2M, bidirectional)
    _RGBD_FUSION_CLS = {
        "se": RGBDCrossAttention,
        "mamba": RGBDMambaFusion,
        "cross_v1": RGBDCrossAttentionV1,
        "cross_v2": RGBDCrossAttentionV2,
        "coord_att": RGBDCoordAtt,
        "coord_att_v2": RGBDCoordAttV2,
        "concat": RGBDConcatFusion,
        "add": RGBDAddFusion,
        "old_cmx_ffm": RGBDYOLOv5FRMFFMFusion,
        "yolov5_frm_ffm": RGBDYOLOv5FRMFFMFusion,
        "cmm": RGBDCrossModalMamba,
    }
    _fusion_key = d.get("rgbd_fusion", "se")
    _fusion_cls = _RGBD_FUSION_CLS.get(_fusion_key, RGBDCrossAttention)
    _p3_wavelet_guided = bool(d.get("p3_wavelet_guided", False))
    _p3_wavelet_hf = bool(d.get("p3_wavelet_HF", False))
    _p3_wavelet_adaptive = bool(d.get("p3_wavelet_adaptive", False))
    _p4_wavelet_guided = bool(d.get("p4_wavelet_guided", False))
    _adaptive_gate = bool(d.get("modality_adaptive_gate", False))
    _rgbd_reliability_gate = bool(d.get("rgbd_reliability_gate", False))
    _rgbd_reliability_mode = d.get("rgbd_reliability_mode", "relative_soft")
    _rgbd_aux_loss = bool(d.get("rgbd_aux_loss", False))
    _rgbd_depth_aux_loss = _rgbd_aux_loss and bool(d.get("rgbd_depth_aux_loss", True))
    _fusion_kw = {}
    if _adaptive_gate:
        _fusion_kw.update(adaptive_gate=True, gate_mode=d.get("gate_mode", "channel"))
    if "fusion_base_scale" in d:
        _fusion_kw["base_scale"] = float(d.get("fusion_base_scale"))
    if "fusion_out_scale" in d:
        _fusion_kw["out_scale"] = float(d.get("fusion_out_scale"))
    if bool(d.get("fusion_learnable_blend", False)):
        _fusion_kw["learnable_blend"] = True
        _fusion_kw["blend_init"] = float(d.get("fusion_blend_init", 0.5))
    if verbose:
        LOGGER.info(
            f"RGB-D fusion mode: '{_fusion_key}' → {_fusion_cls.__name__}"
            + (" (modality adaptive gate)" if _adaptive_gate else "")
        )
    ch_p3, ch_p4, ch_p5 = ch[4], ch[6], ch[10]
    _depth_fpn = bool(d.get("depth_fpn", False))
    if _depth_fpn:
        layers.append(DepthLightFPN(ch_p3, ch_p4, ch_p5))
        if verbose:
            _dfpn = layers[-1]
            LOGGER.info(f"Depth LightFPN enabled ({sum(p.numel() for p in _dfpn.parameters()):,} params)")
    _WAVELET_GUIDED_MAP = {
        "coord_att": RGBDWaveletGuidedCoordAtt,
        "coord_att_v2": RGBDWaveletGuidedCoordAttV2,
    }
    _wavelet_guided_cls = _WAVELET_GUIDED_MAP.get(_fusion_key)
    _p4_fusion_cls = _wavelet_guided_cls if _p4_wavelet_guided and _wavelet_guided_cls else _fusion_cls
    if _p3_wavelet_adaptive:
        _p3_module = RGBDWaveletAdaptiveFusion(
            ch_p3, num_heads=8, kv_pool=10, base_fusion_cls=_fusion_cls, **_fusion_kw
        )
    elif _p3_wavelet_hf:
        _p3_module = RGBDWaveletHFFusion(ch_p3, num_heads=8, kv_pool=10, base_fusion_cls=_fusion_cls, **_fusion_kw)
    else:
        _p3_fusion_cls = _wavelet_guided_cls if _p3_wavelet_guided and _wavelet_guided_cls else _fusion_cls
        _p3_module = _p3_fusion_cls(ch_p3, num_heads=8, kv_pool=10, **_fusion_kw)
    _p4_module = _p4_fusion_cls(ch_p4, num_heads=8, kv_pool=10, **_fusion_kw)
    _p5_module = _fusion_cls(ch_p5, num_heads=8, kv_pool=10, **_fusion_kw)
    layers.append(_p3_module)  # P3 fusion (80x80)
    layers.append(_p4_module)  # P4 fusion (40x40)
    layers.append(_p5_module)  # P5 fusion (20x20)
    if _rgbd_reliability_gate:
        layers.append(RGBDRelativeReliabilityGate(ch_p3, mode=_rgbd_reliability_mode))
        layers.append(RGBDRelativeReliabilityGate(ch_p4, mode=_rgbd_reliability_mode))
        layers.append(RGBDRelativeReliabilityGate(ch_p5, mode=_rgbd_reliability_mode))
        if verbose:
            _gate_params = sum(p.numel() for m in layers[-3:] for p in m.parameters())
            LOGGER.info(f"RGB-D reliability gate enabled: mode={_rgbd_reliability_mode}, {_gate_params:,} params")
    if _rgbd_depth_aux_loss:
        layers.append(RGBDDepthAuxHead(ch_p3))
        if verbose:
            _aux_params = sum(p.numel() for p in layers[-1].parameters())
            LOGGER.info(f"RGB-D depth auxiliary foreground head enabled: {_aux_params:,} params")

    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path) or d.get("scale", "")
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
