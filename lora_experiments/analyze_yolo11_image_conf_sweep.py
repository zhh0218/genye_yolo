#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


CONF_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CLASSES = [0, 1]
MATCH_IOU = 0.50


def parse_thresholds(text):
    return [float(x) for x in text.replace(",", " ").split()]


def empty_summary(cls_id, conf, nms_iou):
    return {
        "total_images": 0,
        "gt_cls_images": 0,
        "pred_cls_images": 0,
        "gt_cls_instances": 0,
        "pred_cls_instances": 0,
        "matched_cls_instances": 0,
        "missed_cls_instances": 0,
        "unmatched_pred_cls_instances": 0,
        "gt_cls_no_pred_images": 0,
        "gt_cls_no_match_images": 0,
        "gt_cls_partial_miss_images": 0,
        "gt_cls_all_matched_images": 0,
        "no_gt_but_pred_cls_images": 0,
        "presence_wrong_images": 0,
        "conf_thres": conf,
        "nms_iou": nms_iou,
        "match_iou": MATCH_IOU,
        "class_id": cls_id,
    }


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes, pred_boxes, thr):
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou >= thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
    return len(used_gt)


def update_summary(summary, gt_count, pred_count, match_count):
    missed = gt_count - match_count
    unmatched_pred = pred_count - match_count
    has_gt = gt_count > 0
    has_pred = pred_count > 0

    summary["total_images"] += 1
    summary["gt_cls_instances"] += gt_count
    summary["pred_cls_instances"] += pred_count
    summary["matched_cls_instances"] += match_count
    summary["missed_cls_instances"] += missed
    summary["unmatched_pred_cls_instances"] += unmatched_pred
    summary["gt_cls_images"] += int(has_gt)
    summary["pred_cls_images"] += int(has_pred)
    summary["gt_cls_no_pred_images"] += int(has_gt and not has_pred)
    summary["gt_cls_no_match_images"] += int(has_gt and match_count == 0)
    summary["gt_cls_partial_miss_images"] += int(has_gt and missed > 0)
    summary["gt_cls_all_matched_images"] += int(has_gt and missed == 0)
    summary["no_gt_but_pred_cls_images"] += int((not has_gt) and has_pred)
    summary["presence_wrong_images"] += int(has_gt != has_pred)


def finalize_summary(summary):
    gt_images = summary["gt_cls_images"]
    hit_images = gt_images - summary["gt_cls_no_pred_images"]
    fp_images = summary["no_gt_but_pred_cls_images"]
    non_cls_images = summary["total_images"] - gt_images

    hit_rate = hit_images / gt_images if gt_images else 0.0
    false_rate = fp_images / (gt_images + fp_images) if gt_images + fp_images else 0.0
    presence_precision = hit_images / (hit_images + fp_images) if hit_images + fp_images else 0.0
    image_f1 = (
        2 * presence_precision * hit_rate / (presence_precision + hit_rate)
        if presence_precision + hit_rate
        else 0.0
    )

    summary["hit_cls_images"] = hit_images
    summary["non_cls_images"] = non_cls_images
    summary["false_positive_cls_images"] = fp_images
    summary["hit_rate"] = hit_rate
    summary["false_rate"] = false_rate
    summary["presence_precision"] = presence_precision
    summary["image_f1"] = image_f1
    summary["gt_cls_image_error_rate_no_match"] = (
        summary["gt_cls_no_match_images"] / gt_images if gt_images else 0.0
    )
    summary["gt_cls_image_error_rate_partial_miss"] = (
        summary["gt_cls_partial_miss_images"] / gt_images if gt_images else 0.0
    )
    summary["instance_recall_at_match_iou"] = (
        summary["matched_cls_instances"] / summary["gt_cls_instances"] if summary["gt_cls_instances"] else 0.0
    )


def build_validator():
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    class ImageConfSweepValidator(SegmentationValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.image_error_summary = {
                f"{conf:.2f}": {
                    str(cls_id): empty_summary(cls_id, conf, float(self.args.iou)) for cls_id in CLASSES
                }
                for conf in CONF_SWEEP
            }

        def _collect_image_errors(self, preds, batch):
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred)

                gt_cls = pbatch["cls"].detach().cpu().tolist()
                gt_boxes = pbatch["bboxes"].detach().cpu().tolist()
                pred_cls = predn["cls"].detach().cpu().tolist()
                pred_boxes = predn["bboxes"].detach().cpu().tolist()
                pred_conf = predn["conf"].detach().cpu().tolist()

                for conf in CONF_SWEEP:
                    conf_key = f"{conf:.2f}"
                    for cls_id in CLASSES:
                        gt = [box for c, box in zip(gt_cls, gt_boxes) if int(c) == cls_id]
                        pr = [
                            box
                            for c, score, box in zip(pred_cls, pred_conf, pred_boxes)
                            if int(c) == cls_id and float(score) >= conf
                        ]
                        match_count = greedy_match(gt, pr, MATCH_IOU)
                        update_summary(self.image_error_summary[conf_key][str(cls_id)], len(gt), len(pr), match_count)

        def update_metrics(self, preds, batch):
            self._collect_image_errors(preds, batch)
            super().update_metrics(preds, batch)

        def finalize_metrics(self):
            super().finalize_metrics()
            rows = []
            for by_class in self.image_error_summary.values():
                for summary in by_class.values():
                    finalize_summary(summary)
            for conf_key, by_class in self.image_error_summary.items():
                for summary in by_class.values():
                    rows.append({"conf": float(conf_key), **summary})

            out_dir = Path(self.save_dir) / "image_error_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary_by_conf_class.json").write_text(
                json.dumps(self.image_error_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (out_dir / "summary_by_conf_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(self.image_error_summary, ensure_ascii=False, indent=2))
            print(f"image_error_analysis={out_dir}")

    return ImageConfSweepValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/workspace/ultralytics-main_for_genye")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--thresholds", default="0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", default="0 1")
    args = parser.parse_args()

    global CONF_SWEEP, CLASSES
    CONF_SWEEP = sorted(parse_thresholds(args.thresholds))
    CLASSES = [int(x) for x in args.classes.replace(",", " ").split()]
    min_conf = min(CONF_SWEEP)

    sys.path.insert(0, args.repo)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.val(
        validator=build_validator(),
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=min_conf,
        iou=args.iou,
        save_txt=False,
        save_json=False,
        plots=False,
        verbose=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
