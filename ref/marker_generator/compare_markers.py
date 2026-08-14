#!/usr/bin/env python3
"""
Compare reconstructed STag markers against ground-truth markers.

For each PNG shared between the generated directory and the ground-truth
directory, this script:
  1. computes a per-pixel difference (on the RGB channels),
  2. writes a side-by-side "diff image" (ground truth | generated | diff),
  3. prints a numeric summary (total / significant diff pixels).

Usage:
    python3 compare_markers.py [--generated DIR] [--truth DIR] [--out DIR] [--threshold N]

Defaults:
    --generated HD23_generated
    --truth     HD23
    --out       HD23_diff
    --threshold 50   (pixels whose max channel diff exceeds this are "significant")
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
import cv2


def read_rgb(path):
    """Read an image as BGR, dropping any alpha channel."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    return img


# layout constants for the side-by-side diff image
BORDER_COLOR = (0, 0, 0)  # black outer border / separators (BGR)
LABEL_BG = (30, 30, 30)  # dark grey label strip background (BGR)
LABEL_FG = (255, 255, 255)  # white label text (BGR)
BORDER_WIDTH = 4  # thickness of the outer border and separators
LABEL_HEIGHT = 40  # height of the caption strip under each panel


def _add_border(img, thickness=BORDER_WIDTH, color=BORDER_COLOR):
    """Draw a rectangular border around an image (in place)."""
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


def _add_label(img, text):
    """Append a caption strip with *text* below the image, returning a new image."""
    w = img.shape[1]
    strip = np.full((LABEL_HEIGHT, w, 3), LABEL_BG, np.uint8)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx = (w - tw) // 2
    ty = (LABEL_HEIGHT + th) // 2
    cv2.putText(
        strip, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_FG, 2, cv2.LINE_AA
    )
    return np.vstack([img, strip])


def build_diff_visual(truth, generated, missing_mask, extra_mask, title=None):
    """Return a labelled side-by-side image:
    ground truth | generated | diff highlight, each with a caption.

    The diff panel colours mismatches by semantic:
      - RED   = "missing" : black in truth, white in generated (false negative)
      - BLUE  = "extra"   : white in truth, black in generated (false positive)

    Panels are separated by vertical black bars, the whole composite is framed
    by a black border, and an optional *title* strip is placed on top.
    """
    # diff highlight: grayscale copy of truth, with mismatches recoloured
    vis = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[missing_mask] = (0, 0, 255)  # BGR red   -> missing
    vis[extra_mask] = (255, 0, 0)  # BGR blue  -> extra

    # caption under each panel
    panels = [
        _add_label(truth, "Ground Truth"),
        _add_label(generated, "Generated"),
        _add_label(vis, "Diff: red=missing, blue=extra"),
    ]

    # vertical separator bars between panels
    sep = np.full((panels[0].shape[0], BORDER_WIDTH, 3), BORDER_COLOR, np.uint8)
    composite = panels[0]
    for p in panels[1:]:
        composite = np.hstack([composite, sep, p])

    # optional title strip on top
    if title:
        tw_title = composite.shape[1]
        strip = np.full((LABEL_HEIGHT, tw_title, 3), LABEL_BG, np.uint8)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            strip,
            title,
            ((tw_title - tw) // 2, (LABEL_HEIGHT + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            LABEL_FG,
            2,
            cv2.LINE_AA,
        )
        composite = np.vstack([strip, composite])

    _add_border(composite)
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Compare STag markers to ground truth."
    )
    parser.add_argument(
        "--generated",
        default="HD23_generated",
        help="directory of generated markers (default: HD23_generated)",
    )
    parser.add_argument(
        "--truth",
        default="HD23",
        help="directory of ground-truth markers (default: HD23)",
    )
    parser.add_argument(
        "--out",
        default="HD23_diff",
        help="output directory for diff images (default: HD23_diff)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="significant diff threshold on max channel diff (default: 50)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.generated):
        raise SystemExit(f"generated directory not found: {args.generated}")
    if not os.path.isdir(args.truth):
        raise SystemExit(f"truth directory not found: {args.truth}")

    os.makedirs(args.out, exist_ok=True)

    truth_files = sorted(
        f for f in os.listdir(args.truth) if f.lower().endswith(".png")
    )
    gen_files = sorted(
        f for f in os.listdir(args.generated) if f.lower().endswith(".png")
    )
    common = [f for f in truth_files if f in set(gen_files)]

    if not common:
        raise SystemExit("no common PNG files between the two directories")

    print(f"comparing {len(common)} markers")
    print(f"{'file':>12}  {'total diff':>12}  {'significant':>12}  {'pct diff':>9}")
    print("-" * 52)

    total_all = 0
    sig_all = 0
    # The progress bar must share the stream with the per-file result lines, and
    # result lines are emitted via pbar.write(), otherwise tqdm (stderr) and
    # print (stdout) interleave chaotically on the terminal.
    pbar = tqdm(common, desc="Comparing", unit="marker", leave=False, file=sys.stdout)
    for f in pbar:
        truth = read_rgb(os.path.join(args.truth, f))
        generated = read_rgb(os.path.join(args.generated, f))

        if truth.shape != generated.shape:
            pbar.write(
                f"  ! {f}: size mismatch {truth.shape} vs {generated.shape}, skipped"
            )
            continue

        diff = cv2.absdiff(truth, generated)
        diff_mask = diff.sum(axis=2) > 0
        max_channel = diff.max(axis=2)
        sig_mask = max_channel > args.threshold

        # classify mismatches by semantic (marker pixels are black on white)
        tg = truth[:, :, 0]  # truth grayscale (BGR images: any channel works)
        gg = generated[:, :, 0]
        missing = diff_mask & (tg == 0) & (gg == 255)  # truth black, gen white
        extra = diff_mask & (tg == 255) & (gg == 0)  # truth white, gen black

        n_total = int(diff_mask.sum())
        n_sig = int(sig_mask.sum())
        n_missing = int(missing.sum())
        n_extra = int(extra.sum())
        pct = 100.0 * n_total / diff_mask.size
        total_all += n_total
        sig_all += n_sig

        pbar.write(
            f"{f:>12}  {n_total:>12d}  {n_sig:>12d}  {pct:>8.2f}%"
            f"  (missing {n_missing}, extra {n_extra})"
        )

        combo = build_diff_visual(truth, generated, missing, extra, title=f"Marker {f}")
        cv2.imwrite(os.path.join(args.out, f), combo)

    print("-" * 52)
    print(f"{'TOTAL':>12}  {total_all:>12d}  {sig_all:>12d}")
    print(f"\ndiff images written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
