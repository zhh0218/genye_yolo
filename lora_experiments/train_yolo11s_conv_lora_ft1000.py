#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conv-LoRA fine-tuning for the custom YOLOv11s RGB-D segmentation model.

The experiment keeps the existing YAML/model code untouched:
1. load the current best YOLOv11s RGB-D checkpoint;
2. insert low-rank Conv-LoRA branches into RGB/depth backbone and neck Conv2d;
3. freeze original backbone/neck weights;
4. train LoRA params plus Segment head and RGB-D fusion modules;
5. export merged checkpoints that can be loaded as normal YOLO weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim


REPO = Path("/workspace/ultralytics-main_for_genye")
BASE_WEIGHTS = REPO / "YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt"
FT1000_YAML = Path("/workspace/Datasets/genye_ft_splits_seed20260526_v2/yolo11/ft1000.yaml")
PROJECT = REPO / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora"


class Conv2dLoRA(nn.Module):
    """Residual low-rank adapter for Conv2d.

    For a base Conv2d W, the forward pass is:
        y = W(x) + scale * B(A(x))
    where A is a rank-r convolution and B is a 1x1 projection.
    """

    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("Conv2dLoRA only supports groups=1 in this experiment.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)

        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling

    @torch.no_grad()
    def merged_conv(self) -> nn.Conv2d:
        base = self.base
        merged = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=base.bias is not None,
            padding_mode=base.padding_mode,
        ).to(device=base.weight.device, dtype=base.weight.dtype)
        down = self.lora_down.weight.detach().float()
        up = self.lora_up.weight.detach().float()[:, :, 0, 0]
        delta = torch.einsum("or,rijk->oijk", up, down) * self.scaling
        merged.weight.copy_((base.weight.detach().float() + delta).to(dtype=base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias.detach())
        return merged


def parse_layer_ids(spec: str) -> set[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def inject_lora(module: nn.Module, rank: int, alpha: float, min_channels: int, prefix: str = "") -> list[dict]:
    records: list[dict] = []
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Conv2dLoRA):
            continue
        if isinstance(child, nn.Conv2d):
            ok = child.groups == 1 and min(child.in_channels, child.out_channels) >= min_channels
            if ok:
                setattr(module, name, Conv2dLoRA(child, rank=rank, alpha=alpha))
                records.append(
                    {
                        "name": child_prefix,
                        "in_channels": child.in_channels,
                        "out_channels": child.out_channels,
                        "kernel_size": list(child.kernel_size),
                        "stride": list(child.stride),
                    }
                )
            continue
        records.extend(inject_lora(child, rank, alpha, min_channels, child_prefix))
    return records


def inject_lora_to_model(model: nn.Module, target_layers: set[int], rank: int, alpha: float, min_channels: int) -> list[dict]:
    records: list[dict] = []
    layers = getattr(model, "model", None)
    if layers is None:
        raise RuntimeError("Expected Ultralytics model with .model Sequential layers.")
    for idx in sorted(target_layers):
        if idx < 0 or idx >= len(layers):
            continue
        records.extend(inject_lora(layers[idx], rank, alpha, min_channels, prefix=f"model.{idx}"))
    return records


def apply_trainable_policy(model: nn.Module, trainable_layers: set[int]) -> dict:
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_prefixes = tuple(f"model.{idx}." for idx in sorted(trainable_layers))
    trainable_params = 0
    total_params = 0
    trainable_names = []
    for name, p in model.named_parameters():
        total_params += p.numel()
        allow = "lora_down." in name or "lora_up." in name or name.startswith(trainable_prefixes)
        if ".dfl." in name:
            allow = False
        p.requires_grad = bool(allow)
        if p.requires_grad:
            trainable_params += p.numel()
            trainable_names.append(name)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / max(total_params, 1),
        "trainable_name_count": len(trainable_names),
        "trainable_name_examples": trainable_names[:40],
    }


def merge_lora_inplace(module: nn.Module) -> int:
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Conv2dLoRA):
            setattr(module, name, child.merged_conv())
            merged += 1
        else:
            merged += merge_lora_inplace(child)
    return merged


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_checkpoint(src: Path, dst: Path) -> dict:
    ckpt = torch.load(src, map_location="cpu")
    merged_counts = {}
    for key in ("model", "ema"):
        obj = ckpt.get(key)
        if isinstance(obj, nn.Module):
            merged_counts[key] = merge_lora_inplace(obj)
    ckpt["lora_merged_from"] = str(src)
    ckpt["lora_merged_counts"] = merged_counts
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return merged_counts


def backup_and_merge_checkpoint_inplace(src: Path) -> dict:
    backup = src.with_name(f"{src.stem}_lora{src.suffix}")
    if not backup.exists():
        shutil.copy2(src, backup)
    return merge_checkpoint(src, src)


def make_trainer(args):
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import LOGGER, RANK, colorstr
    from ultralytics.engine.trainer import MuSGD

    target_layers = parse_layer_ids(args.lora_layers)
    trainable_layers = parse_layer_ids(args.trainable_layers)

    class LoRASegmentationTrainer(SegmentationTrainer):
        lora_records: list[dict] = []
        lora_summary: dict = {}

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
            if weights:
                model.load(weights)
            self.lora_records = inject_lora_to_model(
                model,
                target_layers=target_layers,
                rank=args.rank,
                alpha=args.alpha,
                min_channels=args.min_channels,
            )
            self.lora_summary = apply_trainable_policy(model, trainable_layers=trainable_layers)
            LOGGER.info(
                f"Conv-LoRA injected into {len(self.lora_records)} Conv2d modules; "
                f"trainable {self.lora_summary['trainable_params']:,}/"
                f"{self.lora_summary['total_params']:,} "
                f"({self.lora_summary['trainable_ratio'] * 100:.2f}%)."
            )
            return model

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            self.lora_summary = apply_trainable_policy(model, trainable_layers=trainable_layers)
            summary = {
                **self.lora_summary,
                "rank": args.rank,
                "alpha": args.alpha,
                "min_channels": args.min_channels,
                "lora_layers": sorted(target_layers),
                "trainable_layers": sorted(trainable_layers),
                "lora_module_count": len(self.lora_records),
                "lora_records": self.lora_records,
            }
            save_json(Path(self.save_dir) / "conv_lora_summary.json", summary)

            g = [], [], []
            bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
            if name == "auto":
                LOGGER.info(
                    f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                    f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                    "determining best optimizer automatically..."
                )
                nc = self.data.get("nc", 10)
                lr_fit = round(0.002 * 5 / (4 + nc), 6)
                name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
                self.args.warmup_bias_lr = 0.0

            use_muon = name.lower() == "musgd"
            g_decay_sgd = []
            for module_name, module in model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    if "bias" in fullname:
                        g[2].append(param)
                    elif isinstance(module, bn) or "logit_scale" in fullname:
                        g[1].append(param)
                    else:
                        if use_muon and param.ndim not in {2, 4}:
                            g_decay_sgd.append(param)
                        else:
                            g[0].append(param)

            optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
            name = {x.lower(): x for x in optimizers}.get(name.lower())
            if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
                optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
            elif name == "RMSProp":
                optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
            elif name == "SGD":
                optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
            elif name == "MuSGD":
                optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
                g_musgd = [
                    {"params": g[2], **optim_args, "weight_decay": 0.0, "use_muon": False, "param_group": "bias"},
                    {"params": g[1], **optim_args, "weight_decay": 0.0, "use_muon": False, "param_group": "bn"},
                    {"params": g[0], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"},
                    {"params": g_decay_sgd, **optim_args, "weight_decay": decay, "use_muon": False, "param_group": "sgd"},
                ]
                optimizer = MuSGD(params=g_musgd, muon=0.5, sgd=0.5)
                LOGGER.info(
                    f"{colorstr('optimizer:')} MuSGD(lr={lr}, momentum={momentum}) with trainable-only groups "
                    f"{len(g[1])} bn, {len(g[0])} muon, {len(g_decay_sgd)} sgd, {len(g[2])} bias"
                )
                return optimizer
            else:
                raise NotImplementedError(f"Optimizer '{name}' not found in {optimizers}.")

            optimizer.add_param_group({"params": g[0], "weight_decay": decay})
            optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
            LOGGER.info(
                f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) "
                f"with trainable-only groups {len(g[1])} no_decay, {len(g[0])} decay, {len(g[2])} bias"
            )
            return optimizer

        def final_eval(self):
            merged = {}
            for src in {Path(self.last), Path(self.best)}:
                if src.exists():
                    merged[src.name] = backup_and_merge_checkpoint_inplace(src)
            save_json(Path(self.save_dir) / "conv_lora_pre_final_eval_merge.json", merged)
            return super().final_eval()

    return LoRASegmentationTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(BASE_WEIGHTS))
    parser.add_argument("--data", default=str(FT1000_YAML))
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--name", default="genye_ft1000_yolo11s_conv_lora_r8")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--min-channels", type=int, default=8)
    parser.add_argument("--lora-layers", default="0-22,24-35")
    parser.add_argument("--trainable-layers", default="23,36-38")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO))

    from ultralytics import YOLO

    if args.dry_run:
        model = YOLO(args.weights).model
        records = inject_lora_to_model(
            model,
            target_layers=parse_layer_ids(args.lora_layers),
            rank=args.rank,
            alpha=args.alpha,
            min_channels=args.min_channels,
        )
        summary = apply_trainable_policy(model, parse_layer_ids(args.trainable_layers))
        print(json.dumps({**summary, "lora_module_count": len(records), "records": records[:20]}, ensure_ascii=False, indent=2))
        return

    model = YOLO(args.weights)
    trainer_cls = make_trainer(args)
    model.train(
        trainer=trainer_cls,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        amp=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0001,
        plots=False,
        val=True,
    )

    save_dir = Path(args.project) / args.name
    merged = {}
    for stem in ("best", "last"):
        src = save_dir / "weights" / f"{stem}.pt"
        if src.exists():
            dst = save_dir / "weights" / f"{stem}_merged.pt"
            merged[stem] = merge_checkpoint(src, dst)
    save_json(save_dir / "conv_lora_merged_summary.json", merged)
    print(f"lora_train_done={save_dir}")
    print(f"merged={json.dumps(merged, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
