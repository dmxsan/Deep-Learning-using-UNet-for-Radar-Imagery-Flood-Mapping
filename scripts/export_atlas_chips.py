"""Export representative 128x128 Sentinel-1 flood chips for the portfolio ChipInspector.

Usage:
    python scripts/export_atlas_chips.py
    python scripts/export_atlas_chips.py --data-dir data --threshold 0.5
    python scripts/export_atlas_chips.py --chips 1_s21_training__100_128.tif 1_s21_training__200_128.tif 1_s21_training__300_128.tif
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform as warp_transform
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    MaxPool2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model


MINX = -70.0
MAXX = 70.0
CHIP_SIZE = 128

# Target water coverage brackets (ground-truth %) for automatic 3-chip selection:
# (chip_id, short_label, min_gt_pct, max_gt_pct, ideal_gt_pct)
CHIP_SPECS: list[tuple[str, str, float, float, float]] = [
    ("chip-01", "Main River Channel", 20.0, 38.0, 29.0),
    ("chip-02", "Broad Floodplain", 40.0, 65.0, 52.0),
    ("chip-03", "Tributary & Scattered Water", 10.0, 25.0, 17.5),
]


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Custom IoU metric matching Notebook/unet2122.ipynb."""

    def _f(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        intersection = (yt * yp).sum()
        union = yt.sum() + yp.sum() - intersection
        val = (intersection + 1e-15) / (union + 1e-15)
        return np.asarray(val, dtype=np.float32)

    return tf.numpy_function(_f, [y_true, y_pred], tf.float32)


def conv_block(x: tf.Tensor, num_filters: int) -> tf.Tensor:
    """Double 3x3 Conv + BatchNorm + ReLU block from unet2122.ipynb."""
    for _ in range(2):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def build_model() -> Model:
    """Rebuild the exact U-Net architecture from Notebook/unet2122.ipynb."""
    num_filters = [16, 32, 48, 64]
    inputs = Input((CHIP_SIZE, CHIP_SIZE, 3))

    skip_x = []
    x = inputs
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    x = conv_block(x, num_filters[-1])

    for i, f in enumerate(reversed(num_filters)):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[-(i + 1)]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    return Model(inputs, x)


def load_unet_model(model_path: Path) -> Model:
    """Load the trained .h5 model, falling back to build_model() + load_weights()."""
    try:
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"iou": iou_metric},
            compile=False,
        )
        print(f"Loaded full model from {model_path}")
        return model
    except Exception as exc:
        print(f"load_model() failed ({exc}); rebuilding architecture and loading weights...")
        model = build_model()
        model.load_weights(str(model_path))
        print(f"Loaded weights into rebuilt U-Net from {model_path}")
        return model


def load_test_split(data_dir: Path, split: float = 0.1) -> tuple[list[Path], list[Path]]:
    """Reproduce the exact test_x, test_y split (random_state=42) from unet2122.ipynb."""
    images = sorted((data_dir / "train").glob("*"))
    masks = sorted((data_dir / "train_masks").glob("*"))
    if not images or len(images) != len(masks):
        raise RuntimeError(
            f"Expected equal non-empty train/ and train_masks/ in {data_dir}, "
            f"got {len(images)} images and {len(masks)} masks."
        )

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, _ = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, _ = train_test_split(masks, test_size=valid_size, random_state=42)

    _, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    _, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
    return list(test_x), list(test_y)


def read_sar_chip(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Read a 3-band 128x128 SAR GeoTIFF.

    Returns:
        raw_hwc: (128, 128, 3) float32 array in original dB/linear units.
        norm_hwc: (128, 128, 3) float32 array normalized with MINX/MAXX.
        coords_str: Formatted center lat/lon or fallback filename label.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (C, H, W)
        raw_hwc = np.moveaxis(arr, 0, -1)
        coords_str = format_center_coords(src, path.name)

    norm_hwc = ((raw_hwc - MINX) / (MAXX - MINX)).astype(np.float32)
    return raw_hwc, norm_hwc, coords_str


def format_center_coords(src: rasterio.DatasetReader, filename: str) -> str:
    """Format the GeoTIFF center coordinate as e.g. '6.97° S, 110.42° E'."""
    try:
        if src.crs is None or src.transform.is_identity:
            return f"{filename} · {CHIP_SIZE}×{CHIP_SIZE}"
        cx, cy = src.xy(src.height / 2.0, src.width / 2.0)
        if not src.crs.is_geographic:
            xs, ys = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
            lon, lat = float(xs[0]), float(ys[0])
        else:
            lon, lat = float(cx), float(cy)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return f"{filename} · {CHIP_SIZE}×{CHIP_SIZE}"
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"{abs(lat):.2f}° {lat_dir}, {abs(lon):.2f}° {lon_dir}"
    except Exception:
        return f"{filename} · {CHIP_SIZE}×{CHIP_SIZE}"


def read_mask_chip(path: Path) -> np.ndarray:
    """Read ground-truth mask TIFF and map class 2 -> 1 as in unet2122.ipynb."""
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.int16)
    mask[mask == 2] = 1
    return (mask > 0).astype(np.uint8)


def compute_binary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute binary IoU between 0/1 uint8 arrays."""
    intersection = float(np.logical_and(gt > 0, pred > 0).sum())
    union = float(np.logical_or(gt > 0, pred > 0).sum())
    if union == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return intersection / union


def render_sar_rgb(raw_hwc: np.ndarray) -> Image.Image:
    """Render 3-band SAR chip cleanly with 2nd-98th percentile stretch per band."""
    out = np.zeros_like(raw_hwc, dtype=np.uint8)
    for b in range(raw_hwc.shape[-1]):
        band = raw_hwc[:, :, b]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        p2, p98 = np.percentile(valid, [2.0, 98.0])
        if p98 <= p2:
            p2, p98 = float(valid.min()), float(valid.max())
        if p98 > p2:
            scaled = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            out[:, :, b] = (scaled * 255.0).round().astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def render_binary_mask(mask_hw: np.ndarray) -> Image.Image:
    """Render 0/1 mask as 8-bit single-channel PNG (0=dry, 255=flood)."""
    arr = (mask_hw > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def save_preview_strip(
    sar_img: Image.Image,
    gt_img: Image.Image,
    pred_img: Image.Image,
    out_path: Path,
    chip_id: str,
    iou_val: float,
    gt_pct: float,
    pred_pct: float,
) -> None:
    """Save a 2x-upscaled side-by-side SAR | Ground Truth | U-Net Pred strip."""
    scale = 2
    panel = CHIP_SIZE * scale
    gap = 6
    header_h = 24
    width = panel * 3 + gap * 2
    height = header_h + panel

    canvas = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)

    sar_up = sar_img.resize((panel, panel), Image.Resampling.NEAREST)
    gt_up = gt_img.convert("RGB").resize((panel, panel), Image.Resampling.NEAREST)
    pred_up = pred_img.convert("RGB").resize((panel, panel), Image.Resampling.NEAREST)

    canvas.paste(sar_up, (0, header_h))
    canvas.paste(gt_up, (panel + gap, header_h))
    canvas.paste(pred_up, ((panel + gap) * 2, header_h))

    draw.text((6, 5), f"{chip_id} SAR", fill=(230, 235, 240))
    draw.text((panel + gap + 6, 5), f"GT ({gt_pct:.1f}%)", fill=(230, 235, 240))
    draw.text(
        ((panel + gap) * 2 + 6, 5),
        f"Pred ({pred_pct:.1f}% | IoU {iou_val:.2f})",
        fill=(120, 220, 255),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)


def evaluate_candidates(
    model: Model,
    pairs: list[tuple[Path, Path]],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run batched inference on candidate (image, mask) pairs and compute metrics."""
    valid_records: list[dict[str, Any]] = []
    batch_imgs: list[np.ndarray] = []
    batch_meta: list[tuple[Path, np.ndarray, np.ndarray, str]] = []

    def _flush() -> None:
        if not batch_imgs:
            return
        preds = model.predict(np.stack(batch_imgs, axis=0), verbose=0)[..., 0]
        for prob, (img_path, raw_hwc, gt_hw, coords_str) in zip(preds, batch_meta):
            gt_pct = float(gt_hw.mean() * 100.0)
            pred_05 = (prob > 0.5).astype(np.uint8)
            pred_09 = (prob > 0.9).astype(np.uint8)
            valid_records.append(
                {
                    "img_path": img_path,
                    "raw_hwc": raw_hwc,
                    "gt_hw": gt_hw,
                    "prob": prob,
                    "coords": coords_str,
                    "gt_pct": gt_pct,
                    "pred_05_pct": float(pred_05.mean() * 100.0),
                    "pred_09_pct": float(pred_09.mean() * 100.0),
                    "iou_05": compute_binary_iou(gt_hw, pred_05),
                    "iou_09": compute_binary_iou(gt_hw, pred_09),
                }
            )
        batch_imgs.clear()
        batch_meta.clear()

    for img_path, mask_path in pairs:
        raw_hwc, norm_hwc, coords_str = read_sar_chip(img_path)
        if raw_hwc.shape != (CHIP_SIZE, CHIP_SIZE, 3):
            continue
        if not np.isfinite(raw_hwc).all() or float(raw_hwc.std()) < 1e-4:
            continue
        gt_hw = read_mask_chip(mask_path)
        if gt_hw.shape != (CHIP_SIZE, CHIP_SIZE):
            continue
        batch_imgs.append(norm_hwc)
        batch_meta.append((img_path, raw_hwc, gt_hw, coords_str))
        if len(batch_imgs) >= batch_size:
            _flush()

    _flush()
    return valid_records


def select_representative_chips(
    records: list[dict[str, Any]],
    threshold: float,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Select 3 visually distinct, high-IoU chips across the target flood brackets."""
    iou_key = "iou_05" if threshold <= 0.7 else "iou_09"
    used_names: set[str] = set()
    selected: list[tuple[str, str, dict[str, Any]]] = []

    for chip_id, label_suffix, lo, hi, ideal in CHIP_SPECS:
        pool = [
            r
            for r in records
            if r["img_path"].name not in used_names and lo <= r["gt_pct"] <= hi
        ]
        if not pool:
            # Fallback to nearest coverage if strict bracket is empty
            pool = [
                r
                for r in records
                if r["img_path"].name not in used_names and 2.0 <= r["gt_pct"] <= 95.0
            ]
        # Rank by high IoU, with slight tie-breaker toward ideal coverage
        best = max(
            pool,
            key=lambda r: r[iou_key] - 0.002 * abs(r["gt_pct"] - ideal),
        )
        used_names.add(best["img_path"].name)
        selected.append((chip_id, label_suffix, best))

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export U-Net flood chips for ChipInspector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root containing train/ and train_masks/.")
    parser.add_argument("--model", type=Path, default=Path("Notebook/unet_flood.h5"), help="Path to trained Keras .h5 model.")
    parser.add_argument("--out-dir", type=Path, default=Path("exports/radar-flood-unet"), help="Web asset output directory.")
    parser.add_argument("--preview-dir", type=Path, default=Path("exports/_previews"), help="Side-by-side preview output directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Prediction threshold (default: auto-select cleaner of 0.5 vs 0.9).")
    parser.add_argument("--chips", nargs="*", default=None, help="Optional explicit filenames from train/ to export.")
    parser.add_argument("--max-eval", type=int, default=300, help="Max test split chips to evaluate during auto-selection.")
    parser.add_argument("--mosaic", action="store_true", help="Mosaic contiguous KxK tile blocks instead of single 128x128 tiles.")
    parser.add_argument("--mosaic-grid", type=int, default=4, help="Grid size K for KxK tile mosaic (e.g. 4 -> 512x512, 6 -> 768x768).")
    return parser.parse_args()


def build_scene_grids(data_dir: Path) -> dict[str, dict[tuple[int, int], tuple[Path, Path, float, float, float]]]:
    """Index all train/ + train_masks/ tiles into (row, col) grids per scene prefix."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    pairs: list[tuple[Path, Path]] = []
    for img_p in sorted((data_dir / "train").glob("*.tif")):
        mask_p = data_dir / "train_masks" / f"mask_{img_p.name}"
        if mask_p.exists():
            pairs.append((img_p, mask_p))

    if not pairs:
        raise RuntimeError(f"No .tif tiles found in {data_dir / 'train'}")

    def _scan_one(pair: tuple[Path, Path]) -> tuple[str, Path, Path, float, float, float, float, float]:
        img_p, mask_p = pair
        prefix = img_p.name.split("__")[0]
        with rasterio.open(img_p) as src:
            left, top = src.bounds.left, src.bounds.top
            rx, ry = src.res
        with rasterio.open(mask_p) as msrc:
            m = msrc.read(1)
            water_pct = float(((m == 1) | (m == 2)).mean() * 100.0)
        return prefix, img_p, mask_p, left, top, rx, ry, water_pct

    raw_scenes: dict[str, list[tuple[Path, Path, float, float, float]]] = defaultdict(list)
    res_x = res_y = None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for prefix, img_p, mask_p, left, top, rx, ry, water_pct in ex.map(_scan_one, pairs):
            if res_x is None:
                res_x, res_y = rx, ry
            raw_scenes[prefix].append((img_p, mask_p, left, top, water_pct))

    tile_w = CHIP_SIZE * res_x
    tile_h = CHIP_SIZE * res_y
    grids: dict[str, dict[tuple[int, int], tuple[Path, Path, float, float, float]]] = {}
    for prefix, items in raw_scenes.items():
        min_l = min(x[2] for x in items)
        max_t = max(x[3] for x in items)
        grid: dict[tuple[int, int], tuple[Path, Path, float, float, float]] = {}
        for img_p, mask_p, left, top, water_pct in items:
            col = round((left - min_l) / tile_w)
            row = round((max_t - top) / tile_h)
            grid[(row, col)] = (img_p, mask_p, left, top, water_pct)
        grids[prefix] = grid
    return grids


def find_mosaic_windows(
    grids: dict[str, dict[tuple[int, int], tuple[Path, Path, float, float, float]]],
    k: int,
) -> list[dict[str, Any]]:
    """Find all 100%-filled KxK contiguous windows with 8%-70% water coverage."""
    windows: list[dict[str, Any]] = []
    for prefix, grid in sorted(grids.items()):
        for r, c in grid:
            tiles: list[tuple[int, int, Path, Path]] = []
            water_sum = 0.0
            ok = True
            for dr in range(k):
                for dc in range(k):
                    cell = grid.get((r + dr, c + dc))
                    if cell is None:
                        ok = False
                        break
                    img_p, mask_p, _, _, w_pct = cell
                    tiles.append((dr, dc, img_p, mask_p))
                    water_sum += w_pct
                if not ok:
                    break
            if ok:
                avg_pct = water_sum / (k * k)
                if 8.0 <= avg_pct <= 70.0:
                    windows.append(
                        {
                            "prefix": prefix,
                            "row": r,
                            "col": c,
                            "k": k,
                            "gt_pct": avg_pct,
                            "tiles": tiles,
                        }
                    )
    return windows


def _cosine_blend_window(size: int = CHIP_SIZE) -> np.ndarray:
    """2D raised-cosine weight window to smoothly blend overlapping 128x128 patches."""
    ramp = np.sin(np.pi * (np.arange(size, dtype=np.float32) + 0.5) / size) ** 2
    win = np.outer(ramp, ramp)
    return np.maximum(win, 0.05).astype(np.float32)


def stitch_and_predict_mosaic(
    model: Model,
    window: dict[str, Any],
    threshold: float,
    stride: int = 32,
) -> dict[str, Any]:
    """Stitch KxK tiles and run overlapping sliding-window inference (no TTA)."""
    k = window["k"]
    full_size = k * CHIP_SIZE
    raw_mosaic = np.zeros((full_size, full_size, 3), dtype=np.float32)
    norm_mosaic = np.zeros((full_size, full_size, 3), dtype=np.float32)
    gt_mosaic = np.zeros((full_size, full_size), dtype=np.uint8)
    center_coords = ""

    for dr, dc, img_p, mask_p in window["tiles"]:
        raw_hwc, norm_hwc, coords_str = read_sar_chip(img_p)
        gt_hw = read_mask_chip(mask_p)
        y0, y1 = dr * CHIP_SIZE, (dr + 1) * CHIP_SIZE
        x0, x1 = dc * CHIP_SIZE, (dc + 1) * CHIP_SIZE
        raw_mosaic[y0:y1, x0:x1] = raw_hwc
        norm_mosaic[y0:y1, x0:x1] = norm_hwc
        gt_mosaic[y0:y1, x0:x1] = gt_hw
        if dr == k // 2 and dc == k // 2:
            center_coords = coords_str

    ys = list(range(0, full_size - CHIP_SIZE + 1, stride))
    xs = list(range(0, full_size - CHIP_SIZE + 1, stride))
    if ys[-1] != full_size - CHIP_SIZE:
        ys.append(full_size - CHIP_SIZE)
    if xs[-1] != full_size - CHIP_SIZE:
        xs.append(full_size - CHIP_SIZE)

    patches = np.stack(
        [norm_mosaic[y : y + CHIP_SIZE, x : x + CHIP_SIZE] for y in ys for x in xs],
        axis=0,
    )
    probs = model.predict(patches, batch_size=32, verbose=0)[..., 0]

    weight_win = _cosine_blend_window(CHIP_SIZE)
    prob_acc = np.zeros((full_size, full_size), dtype=np.float32)
    weight_acc = np.zeros((full_size, full_size), dtype=np.float32)
    idx = 0
    for y in ys:
        for x in xs:
            prob_acc[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += probs[idx] * weight_win
            weight_acc[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += weight_win
            idx += 1

    prob_mosaic = prob_acc / np.maximum(weight_acc, 1e-6)

    pred_05 = (prob_mosaic > 0.5).astype(np.uint8)
    pred_09 = (prob_mosaic > 0.9).astype(np.uint8)
    pred_thr = (prob_mosaic > threshold).astype(np.uint8)
    iou_05 = compute_binary_iou(gt_mosaic, pred_05)
    iou_09 = compute_binary_iou(gt_mosaic, pred_09)
    iou_val = compute_binary_iou(gt_mosaic, pred_thr)

    return {
        "raw_hwc": raw_mosaic,
        "gt_hw": gt_mosaic,
        "pred_hw": pred_thr,
        "gt_pct": float(gt_mosaic.mean() * 100.0),
        "pred_pct": float(pred_thr.mean() * 100.0),
        "iou": iou_val,
        "iou_05": iou_05,
        "iou_09": iou_09,
        "coords": center_coords,
        "source_file": f"{window['prefix']} ({k}x{k} mosaic @ r{window['row']}c{window['col']})",
    }


def run_mosaic_export(args: argparse.Namespace, model: Model) -> None:
    """Select 3 non-overlapping KxK flooded mosaics and export them."""
    k = args.mosaic_grid
    print(f"Scanning scene grids for {k}x{k} ({k * CHIP_SIZE}x{k * CHIP_SIZE} px) contiguous mosaics...")
    grids = build_scene_grids(args.data_dir)
    windows = find_mosaic_windows(grids, k)
    if len(windows) < 3 and k > 4:
        print(f"Only {len(windows)} windows for {k}x{k}; falling back to 4x4...")
        k = 4
        windows = find_mosaic_windows(grids, k)
    if not windows:
        raise RuntimeError(f"No contiguous {k}x{k} flooded mosaic windows found.")

    print(f"Found {len(windows)} candidate {k}x{k} flooded windows. Selecting 3 distinct areas...")
    chosen_thr = args.threshold if args.threshold is not None else 0.5
    chosen_mosaics: list[tuple[str, str, dict[str, Any]]] = []
    used_boxes: list[tuple[str, int, int]] = []

    def _overlaps(w: dict[str, Any]) -> bool:
        for pref, r0, c0 in used_boxes:
            if w["prefix"] == pref and abs(w["row"] - r0) < k and abs(w["col"] - c0) < k:
                return True
        return False

    for chip_id, short_name, lo, hi, ideal in CHIP_SPECS:
        pool = [w for w in windows if not _overlaps(w) and lo <= w["gt_pct"] <= hi]
        if not pool:
            pool = [w for w in windows if not _overlaps(w)] or windows
        # Evaluate top 4 closest-to-ideal candidates with the model and pick highest IoU
        pool.sort(key=lambda w: abs(w["gt_pct"] - ideal))
        evaluated = [
            (w, stitch_and_predict_mosaic(model, w, chosen_thr))
            for w in pool[:4]
        ]
        best_w, best_res = max(evaluated, key=lambda pair: pair[1]["iou"])
        used_boxes.append((best_w["prefix"], best_w["row"], best_w["col"]))
        chosen_mosaics.append((chip_id, short_name, best_res))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    samples_manifest: list[dict[str, Any]] = []
    for chip_id, short_name, res in chosen_mosaics:
        sar_img = render_sar_rgb(res["raw_hwc"])
        pred_img = render_binary_mask(res["pred_hw"])
        gt_img = render_binary_mask(res["gt_hw"])

        sar_path = args.out_dir / f"{chip_id}-sar.png"
        pred_path = args.out_dir / f"{chip_id}-pred.png"
        gt_path = args.out_dir / f"{chip_id}-gt.png"
        preview_path = args.preview_dir / f"{chip_id}-preview.png"

        sar_img.save(sar_path, optimize=True)
        pred_img.save(pred_path, optimize=True)
        gt_img.save(gt_path, optimize=True)

        # Save 1x-scale side-by-side preview for large mosaics so it stays readable
        panel = sar_img.width
        gap = 8
        header_h = 28
        canvas = Image.new("RGB", (panel * 3 + gap * 2, header_h + panel), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(sar_img, (0, header_h))
        canvas.paste(gt_img.convert("RGB"), (panel + gap, header_h))
        canvas.paste(pred_img.convert("RGB"), ((panel + gap) * 2, header_h))
        draw.text((8, 6), f"{chip_id} SAR ({panel}x{panel})", fill=(230, 235, 240))
        draw.text((panel + gap + 8, 6), f"GT ({res['gt_pct']:.1f}%)", fill=(230, 235, 240))
        draw.text(
            ((panel + gap) * 2 + 8, 6),
            f"Pred ({res['pred_pct']:.1f}% | IoU {res['iou']:.2f})",
            fill=(120, 220, 255),
        )
        canvas.save(preview_path, optimize=True)

        label_num = chip_id.split("-")[1]
        samples_manifest.append(
            {
                "id": chip_id,
                "label": f"Chip {label_num} · {short_name}",
                "coords": res["coords"],
                "sourceFile": res["source_file"],
                "floodCoveragePct": round(res["pred_pct"], 1),
                "gtCoveragePct": round(res["gt_pct"], 1),
                "iou": round(res["iou"], 2),
                "sar": f"/atlas/radar-flood-unet/{chip_id}-sar.png",
                "pred": f"/atlas/radar-flood-unet/{chip_id}-pred.png",
                "gt": f"/atlas/radar-flood-unet/{chip_id}-gt.png",
            }
        )
        print(
            f"  {chip_id}: {res['source_file']} | GT={res['gt_pct']:.1f}% Pred={res['pred_pct']:.1f}% | "
            f"IoU@0.5={res['iou_05']:.4f} IoU@0.9={res['iou_09']:.4f} | coords={res['coords']}"
        )

    manifest = {
        "plate": "radar-flood-unet",
        "modelFile": "Notebook/unet_flood.h5",
        "threshold": chosen_thr,
        "normRange": [int(MINX), int(MAXX)],
        "samples": samples_manifest,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mosaic manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    model = load_unet_model(args.model)
    if args.mosaic:
        run_mosaic_export(args, model)
        return

    if args.chips:
        pairs = []
        for name in args.chips:
            img_p = args.data_dir / "train" / name
            mask_p = args.data_dir / "train_masks" / f"mask_{name}"
            if not img_p.exists() or not mask_p.exists():
                raise FileNotFoundError(f"Missing image or mask for {name} in {args.data_dir}")
            pairs.append((img_p, mask_p))
    else:
        test_x, test_y = load_test_split(args.data_dir, split=0.1)
        pairs = list(zip(test_x, test_y))[: args.max_eval]

    print(f"Evaluating {len(pairs)} candidate chips...")
    records = evaluate_candidates(model, pairs)
    if not records:
        raise RuntimeError("No valid candidate chips found.")

    # Compare thresholds 0.5 vs 0.9 across non-trivial flooded tiles
    flooded = [r for r in records if 5.0 <= r["gt_pct"] <= 95.0] or records
    mean_iou_05 = float(np.mean([r["iou_05"] for r in flooded]))
    mean_iou_09 = float(np.mean([r["iou_09"] for r in flooded]))
    print(f"Mean IoU on flooded test chips (n={len(flooded)}): thr=0.5 -> {mean_iou_05:.4f} | thr=0.9 -> {mean_iou_09:.4f}")

    chosen_thr = args.threshold
    if chosen_thr is None:
        chosen_thr = 0.5 if mean_iou_05 >= mean_iou_09 else 0.9
    print(f"Using prediction threshold: {chosen_thr}")

    if args.chips:
        chosen = [
            (f"chip-{i + 1:02d}", CHIP_SPECS[i][1] if i < len(CHIP_SPECS) else f"Sample {i + 1}", rec)
            for i, rec in enumerate(records)
        ]
    else:
        chosen = select_representative_chips(records, chosen_thr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    samples_manifest: list[dict[str, Any]] = []
    for chip_id, short_name, rec in chosen:
        prob = rec["prob"]
        gt_hw = rec["gt_hw"]
        pred_hw = (prob > chosen_thr).astype(np.uint8)

        iou_05 = rec["iou_05"]
        iou_09 = rec["iou_09"]
        iou_val = compute_binary_iou(gt_hw, pred_hw)
        pred_pct = float(pred_hw.mean() * 100.0)
        gt_pct = rec["gt_pct"]
        src_name = rec["img_path"].name

        sar_img = render_sar_rgb(rec["raw_hwc"])
        pred_img = render_binary_mask(pred_hw)
        gt_img = render_binary_mask(gt_hw)

        sar_path = args.out_dir / f"{chip_id}-sar.png"
        pred_path = args.out_dir / f"{chip_id}-pred.png"
        gt_path = args.out_dir / f"{chip_id}-gt.png"
        preview_path = args.preview_dir / f"{chip_id}-preview.png"

        sar_img.save(sar_path, optimize=True)
        pred_img.save(pred_path, optimize=True)
        gt_img.save(gt_path, optimize=True)
        save_preview_strip(sar_img, gt_img, pred_img, preview_path, chip_id, iou_val, gt_pct, pred_pct)

        label_num = chip_id.split("-")[1]
        samples_manifest.append(
            {
                "id": chip_id,
                "label": f"Chip {label_num} · {short_name}",
                "coords": rec["coords"],
                "sourceFile": src_name,
                "floodCoveragePct": round(pred_pct, 1),
                "gtCoveragePct": round(gt_pct, 1),
                "iou": round(iou_val, 2),
                "sar": f"/atlas/radar-flood-unet/{chip_id}-sar.png",
                "pred": f"/atlas/radar-flood-unet/{chip_id}-pred.png",
                "gt": f"/atlas/radar-flood-unet/{chip_id}-gt.png",
            }
        )
        print(
            f"  {chip_id}: {src_name} | GT={gt_pct:.1f}% Pred={pred_pct:.1f}% | "
            f"IoU@0.5={iou_05:.4f} IoU@0.9={iou_09:.4f} | coords={rec['coords']}"
        )

    manifest = {
        "plate": "radar-flood-unet",
        "modelFile": "Notebook/unet_flood.h5",
        "threshold": chosen_thr,
        "normRange": [int(MINX), int(MAXX)],
        "samples": samples_manifest,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
