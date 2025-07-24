#!/usr/bin/env python3
"""
edge_register.py

Compare the left edge of each frame with the right edge of the next frame
using both cross-correlation and phase correlation.

Usage
-----
python edge_register.py /path/to/stack  --pattern "*.tif" \
       --edge 20 --channel 0  --out results.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv

import numpy as np
import tifffile                       # fast TIFF + OME-TIFF reader
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.ndimage import fourier_shift

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def load_frames(src: Path, pattern: str) -> np.ndarray:
    """Return a list of 2-channel ndarray frames sorted by name."""
    files = sorted(src.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {src}")
    # read into list → 4-D array: (N, H, W, C)
    frames = []
    for f in files:
        im = tifffile.imread(f)
        # bring to (H, W, C)
        if im.ndim == 3 and im.shape[0] == 2:               # (C, H, W)
            im = np.moveaxis(im, 0, -1)
        if im.ndim != 3 or im.shape[-1] != 2:
            raise ValueError(f"{f} is not 2-channel (got shape {im.shape})")
        frames.append(im.astype(np.float32))
    return np.stack(frames, axis=0)

def normxcorr2(a: np.ndarray, b: np.ndarray) -> float:
    """Return peak of normalized cross-correlation between two same-size images."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)

def edge_strips(frame: np.ndarray, edge: int, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_strip, right_strip) for a single frame, given channel index."""
    left  = frame[:, :edge,  ch]
    right = frame[:, -edge:, ch]
    return left, right

# ------------------------------------------------------------
# main procedure
# ------------------------------------------------------------
def process(frames: np.ndarray, edge: int, ch: int) -> list[dict]:
    results = []
    n = len(frames)
    for i in range(n - 1):
        left_strip , _            = edge_strips(frames[i + 1], edge, ch)
        _          , right_strip2 = edge_strips(frames[i + 0], edge, ch)

        # --- 1. normalized cross-correlation (no sub-pixel) ---
        ncc_score = normxcorr2(left_strip, right_strip2)

        # --- 2. phase correlation (sub-pixel) ---
        shift, pc_error, _ = phase_cross_correlation(
            left_strip,
            right_strip2,
            upsample_factor=100,          # for sub-pixel precision
            normalization=None            # we normalized manually in NCC
        )
        
        combined_strip = np.zeros(shape=(*left_strip.shape[:2], 3), dtype=left_strip.dtype)
        combined_strip[...,0] = left_strip
        combined_strip[...,1:] = np.roll(
            right_strip2, 
            shift=(round(shift[0]), round(shift[1])),
            axis=(0,1)
        )[...,np.newaxis]
        
        to_show = 2*combined_strip/combined_strip.max()
        plt.imshow(to_show[10_000:11_000,...])
        plt.show()

        results.append(
            dict(
                frame_i=i,
                frame_j=i + 1,
                ncc=ncc_score,
                dy=float(shift[0]),        # row shift (+ = down)
                dx=float(shift[1]),        # col shift (+ = right)
                pc_error=float(pc_error),
            )
        )
    return results

def write_csv(rows: list[dict], out_path: Path) -> None:
    keys = rows[0].keys()
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-to-edge registration")
    parser.add_argument("src", type=Path,
                        help="Directory containing image files *or* a multi-page TIFF")
    parser.add_argument("--pattern", default="*.tif",
                        help="Glob for files inside SRC (ignored if SRC is a TIFF)")
    parser.add_argument("--edge", type=int, default=10,
                        help="Width of edge strips (pixels)")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel index to use (0 or 1)")
    parser.add_argument("--out", type=Path,
                        help="Write results CSV to this file")
    args = parser.parse_args()

    # --- load stack ---
    if args.src.is_file():
        # multi-page TIFF
        im = tifffile.imread(args.src)
        if im.ndim == 4:           # (N, C, H, W) or (N, H, W, C)
            if im.shape[1] == 2:   # assume (N,C,H,W)
                im = np.moveaxis(im, 1, -1)
        else:
            raise ValueError(f"Unexpected shape for {args.src}: {im.shape}")
        frames = im.astype(np.float32)
    else:
        frames = load_frames(args.src, args.pattern)

    # --- compute registrations ---
    res = process(frames, args.edge, args.channel)

    # --- print summary ---
    for r in res:
        print(
            f"{r['frame_i']:03d}→{r['frame_j']:03d}  "
            f"NCC={r['ncc']:.4f}  "
            f"Δx={r['dx']:+.3f} px, Δy={r['dy']:+.3f} px  "
            f"(phase corr error={r['pc_error']:.4e})"
        )

    if args.out:
        write_csv(res, args.out)
        print(f"\nWrote {len(res)} rows to {args.out}")
