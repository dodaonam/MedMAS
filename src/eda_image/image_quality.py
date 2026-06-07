from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


QUALITY_COLS = [
    "px_mean",
    "px_std",
    "p01_p99_contrast",
    "pct_near_black",
    "pct_near_white",
    "entropy",
    "sharpness_proxy",
]
CACHE_VERSION = 1


def load_gray(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def ahash_bits(arr_uint8: np.ndarray, size: int = 8) -> str:
    small = Image.fromarray(arr_uint8).resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.array(small, dtype=np.float32)
    bits = (pixels > pixels.mean()).astype(np.uint8).reshape(-1)
    return "".join(bits.astype(str).tolist())


def image_dir_signature(image_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(image_dir.glob("*.png")):
        stat = path.stat()
        hasher.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return hasher.hexdigest()


def _cache_meta(image_dir: Path) -> dict[str, object]:
    image_files = list(image_dir.glob("*.png"))
    return {
        "cache_version": CACHE_VERSION,
        "image_dir": str(image_dir.resolve()),
        "image_count": len(image_files),
        "filename_signature": image_dir_signature(image_dir),
    }


def is_cache_valid(metrics_path: Path, meta_path: Path, image_dir: Path) -> bool:
    if not metrics_path.exists() or not meta_path.exists():
        return False
    try:
        cached = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    expected = _cache_meta(image_dir)
    for key, value in expected.items():
        if cached.get(key) != value:
            return False
    return True


def compute_image_metrics(
    image_names: list[str],
    image_dir: Path,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(image_names):
        path = image_dir / name
        with Image.open(path) as img:
            mode = img.mode
            width, height = img.size
            arr_raw = np.array(img)

        if arr_raw.ndim == 3:
            gray = arr_raw[..., 0].astype(np.float32)
            rgba_flag = int(arr_raw.shape[-1] == 4)
        else:
            gray = arr_raw.astype(np.float32)
            rgba_flag = 0

        gray_u8 = gray.astype(np.uint8)
        p01, p99 = np.percentile(gray, [1, 99])
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        hist = hist + 1e-12
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        rows.append(
            (
                name,
                mode,
                width,
                height,
                rgba_flag,
                float(gray.min()),
                float(gray.max()),
                float(gray.mean()),
                float(gray.std()),
                float(p99 - p01),
                float((gray <= 1).mean()),
                float((gray >= 254).mean()),
                float(-(hist * np.log2(hist)).sum()),
                float(gx.var() + gy.var()),
                ahash_bits(gray_u8, size=8),
            )
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(f"Processed {i + 1}/{len(image_names)} images")

    return pd.DataFrame(
        rows,
        columns=[
            "Image Index",
            "mode",
            "width",
            "height",
            "is_rgba",
            "px_min",
            "px_max",
            "px_mean",
            "px_std",
            "p01_p99_contrast",
            "pct_near_black",
            "pct_near_white",
            "entropy",
            "sharpness_proxy",
            "ahash",
        ],
    )


def compute_or_load_image_metrics(
    image_names: list[str],
    image_dir: Path,
    metrics_path: Path,
    meta_path: Path,
    force: bool = False,
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = (not force) and is_cache_valid(metrics_path, meta_path, image_dir)

    if cache_hit:
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = compute_image_metrics(image_names, image_dir, progress_every=progress_every)
        metrics.to_csv(metrics_path, index=False)
        meta = _cache_meta(image_dir)
        meta["generated_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))

    status = pd.DataFrame(
        [
            ("cache_hit", bool(cache_hit)),
            ("rows", int(len(metrics))),
            ("image_dir", str(image_dir)),
            ("metrics_path", str(metrics_path)),
            ("meta_path", str(meta_path)),
        ],
        columns=["metric", "value"],
    )
    return metrics, status


def compute_rgba_consistency(image_metrics: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    rgba_names = image_metrics.loc[image_metrics["is_rgba"] == 1, "Image Index"].tolist()
    rgb_identical = True
    alpha_constant = True
    for name in rgba_names:
        arr = np.array(Image.open(image_dir / name))
        if arr.ndim == 3 and arr.shape[2] == 4:
            rgb_identical = rgb_identical and bool(
                np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])
            )
            alpha_constant = alpha_constant and bool(len(np.unique(arr[..., 3])) == 1)
    rows = [
        ("rgba_image_count", len(rgba_names)),
        ("rgb_channels_identical_for_rgba", rgb_identical),
        ("alpha_channel_constant_for_rgba", alpha_constant),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_ahash_collisions(
    image_metrics: pd.DataFrame,
    df_img: pd.DataFrame,
) -> tuple[pd.DataFrame, list[list[str]]]:
    groups = [
        group["Image Index"].tolist()
        for _, group in image_metrics.groupby("ahash")
        if len(group) > 1
    ]
    groups = sorted(groups, key=len, reverse=True)
    meta = df_img.set_index("Image Index")
    same_pid_pairs = 0
    cross_pid_pairs = 0
    same_label_pairs = 0
    cross_label_pairs = 0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                same_pid_pairs += int(meta.loc[a, "Patient ID"] == meta.loc[b, "Patient ID"])
                cross_pid_pairs += int(meta.loc[a, "Patient ID"] != meta.loc[b, "Patient ID"])
                same_label_pairs += int(meta.loc[a, "Finding Labels"] == meta.loc[b, "Finding Labels"])
                cross_label_pairs += int(meta.loc[a, "Finding Labels"] != meta.loc[b, "Finding Labels"])

    summary = pd.DataFrame(
        [
            ("collision_groups", len(groups)),
            ("images_in_collision_groups", sum(len(group) for group in groups)),
            ("pairs_same_patient_id", same_pid_pairs),
            ("pairs_cross_patient_id", cross_pid_pairs),
            ("pairs_same_labelset", same_label_pairs),
            ("pairs_cross_labelset", cross_label_pairs),
        ],
        columns=["metric", "value"],
    )
    return summary, groups

