#!/usr/bin/env python3
"""
STag marker generator (Python port of Form1.cs).

Generates STag markers (one PNG per code) for the requested Hamming-distance
families, reading the 48-bit codes from the accompanying ``HD{hd}.txt`` files
and writing the rendered markers into ``HD{hd}_generated/`` subdirectories.

Usage:
    python3 marker_generator.py --HD=all      # generate all families (HD11..HD23)
    python3 marker_generator.py --HD=23       # only HD23
    python3 marker_generator.py --HD=11,15,23 # several families at once

Dependencies:
    numpy, opencv-python, pillow, tqdm
"""

import os
import math

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Marker geometry (mirrors the constants in Form1.cs)
# ---------------------------------------------------------------------------

# number of bits in a marker
NO_OF_BITS = 48

# following values are in ratios, where a marker edge length is 1
BORDER = 0.125  # border width
OUTER_CIRCLE_RADIUS = 0.4  # radius of the circle border
INNER_CIRCLE_RADIUS = 0.35  # radius of the circle that encloses code bits
CODE_RADIUS = (
    0.062482177287080  # radius of the code circles (ratio to innerCircleRadius)
)
FILLER_CODE_RADIUS = 0.7  # radius of the filler code circles (ratio to codeRadius)

# following values are in pixels
FILE_SIZE = 1000

# morphological processing parameters
MORPH_RADIUS = 12
MORPH_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def polar_to_cart(radius, radians):
    """Convert a polar coordinate (radius, angle in radians) to Cartesian
    coordinates in marker space, where (0.5, 0.5) is the marker center and
    the positive Y axis points up."""
    return 0.5 + math.cos(radians) * radius, 0.5 - math.sin(radians) * radius


def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Code layout
# ---------------------------------------------------------------------------


def fill_locs():
    """Compute the 48 code-circle locations and the adjacency list of nearby
    code circles (pairs that should be bridged by filler circles)."""
    code_locs = []

    for i in range(4):
        angle_offset = i * (math.pi / 2)

        code_locs.append(
            polar_to_cart(0.088363142525988, 0.785398163397448 + angle_offset)
        )

        code_locs.append(
            polar_to_cart(0.206935928182607, 0.459275804122858 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(
                0.206935928182607, (math.pi / 2) - 0.459275804122858 + angle_offset
            )
        )

        code_locs.append(
            polar_to_cart(0.313672146827381, 0.200579720495241 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(0.327493143484516, 0.591687617505840 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(
                0.327493143484516, (math.pi / 2) - 0.591687617505840 + angle_offset
            )
        )
        code_locs.append(
            polar_to_cart(
                0.313672146827381, (math.pi / 2) - 0.200579720495241 + angle_offset
            )
        )

        code_locs.append(
            polar_to_cart(0.437421957035861, 0.145724938287167 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(0.437226762361658, 0.433363129825345 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(0.430628029742607, 0.785398163397448 + angle_offset)
        )
        code_locs.append(
            polar_to_cart(
                0.437226762361658, (math.pi / 2) - 0.433363129825345 + angle_offset
            )
        )
        code_locs.append(
            polar_to_cart(
                0.437421957035861, (math.pi / 2) - 0.145724938287167 + angle_offset
            )
        )

    assert len(code_locs) == NO_OF_BITS

    nearby_codes = []
    for i in range(NO_OF_BITS):
        nearby_codes.append(
            [
                j
                for j in range(NO_OF_BITS)
                if i != j and distance(code_locs[i], code_locs[j]) < CODE_RADIUS * 4
            ]
        )

    return code_locs, nearby_codes


# ---------------------------------------------------------------------------
# Image processing (numpy vectorised equivalents of the unsafe pointer loops)
# ---------------------------------------------------------------------------


def _ball_mask(radius):
    """Circular structuring element (excluding the centre), same as
    ``generateBallMask`` in the original."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= radius * radius
    mask[radius, radius] = False  # exclude the centre
    return mask.astype(np.uint8)


def _erode(img, radius, border):
    """Set a pixel to white when more than half of its circular neighbourhood
    (radius ``radius``) is non-black.

    Faithful port of ``erodeBitmap``:
    - only pixels with ``border <= x,y < size - border`` are touched
    - only non-white pixels are candidates
    - a candidate becomes white if the number of non-black pixels in the ball
      mask exceeds ``maskArea / 2`` (integer division).
    """
    mask = _ball_mask(radius)
    mask_area = int(mask.sum())

    gray = img[:, :, 0]
    result = img.copy()

    non_black = (gray > 0).astype(np.float32)
    # number of non-black pixels within the ball mask
    count = cv2.filter2D(non_black, cv2.CV_32F, mask, borderType=cv2.BORDER_CONSTANT)

    non_white = gray < 255
    in_region = _region_mask(img, border)
    to_white = non_white & in_region & (count > mask_area // 2)
    result[to_white] = 255
    return result


def _dilate(img, radius, border):
    """Set a pixel to black when more than half of its circular neighbourhood
    (radius ``radius``) is non-white.

    Faithful port of ``dilateBitmap``:
    - only pixels with ``border <= x,y < size - border`` are touched
    - only non-black pixels are candidates
    - a candidate becomes black if the number of non-white pixels in the ball
      mask exceeds ``maskArea / 2`` (integer division).
    """
    mask = _ball_mask(radius)
    mask_area = int(mask.sum())

    gray = img[:, :, 0]
    result = img.copy()

    non_white = (gray < 255).astype(np.float32)
    count = cv2.filter2D(non_white, cv2.CV_32F, mask, borderType=cv2.BORDER_CONSTANT)

    non_black = gray > 0
    in_region = _region_mask(img, border)
    to_black = non_black & in_region & (count > mask_area // 2)
    result[to_black] = 0
    return result


def _region_mask(img, border):
    """Boolean mask of the pixels touched by the morphological loops, i.e.
    ``border <= x,y < size - border`` (border = int(outerCircleTopLeft))."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), bool)
    m[border : h - border, border : w - border] = True
    return m


def _smooth(img, border):
    """Average the 3x3 neighbourhood of black pixels that have a white
    neighbour, restricted to a circular region inside the outer circle.

    Faithful port of ``smoothBitmap``:
    - region: ``dist_from_center < size/2 - outerCircleTopLeft - 10``
    - candidate: black pixel whose 3x3 neighbourhood contains a white pixel
      in one of the 6 directions (excluding the +x+y and -x-y corners, i.e.
      ``|n1 + n2| < 2`` with n1 the x-offset, n2 the y-offset)
    - value: the mean of the 3x3 neighbourhood (integer truncation).
    """
    h, w = img.shape[:2]
    gray = img[:, :, 0]

    # circular region inside the outer circle
    yy, xx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    in_region = dist_from_center < w / 2 - border - 10

    is_black = gray == 0

    # sum of the 3x3 neighbourhood (BORDER_CONSTANT -> 0 outside)
    kernel = np.ones((3, 3), np.float32)
    total = cv2.filter2D(
        gray.astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT
    )
    avg = (total / 9).astype(np.uint8)  # truncation matches the C# cast to byte

    # "has a white neighbour with |n1+n2| < 2": all 3x3 offsets except the
    # (+1,+1) and (-1,-1) corners
    has_white = np.zeros((h, w), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx + dy) >= 2:  # exclude (+1,+1) and (-1,-1)
                continue
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            has_white |= shifted == 255

    to_smooth = is_black & has_white & in_region

    result = img.copy()
    val = np.clip(avg, 0, 255).astype(np.uint8)
    for c in range(3):
        ch = result[:, :, c].copy()
        ch[to_smooth] = val[to_smooth]
        result[:, :, c] = ch
    return result


# ---------------------------------------------------------------------------
# Anti-aliased ellipse rendering (Pillow supersampling)
# ---------------------------------------------------------------------------

# supersampling factor: render at this multiple resolution, then box-downsample.
# 4x4 gives 16 sub-samples per pixel, a good approximation of GDI+'s area
# coverage anti-aliasing used by FillEllipse with AntiAlias/HighQuality.
_SS = 4


def _fill_ellipse_aa(img, left, top, width, height, color):
    """Fill an ellipse (GDI+ ``FillEllipse`` semantics: bounding box
    ``[left, top, left+width, top+height]``, centre at its mid-point) with
    high-quality anti-aliasing via supersampling.

    ``img`` is a numpy array mutated in place — either HxW (single channel,
    ``color`` is a scalar) or HxWx3 (``color`` is an RGB tuple).
    ``left``/``top``/``width``/``height`` are floats (as in GDI+).
    """
    if width <= 0 or height <= 0:
        return

    cx = left + width / 2.0
    cy = top + height / 2.0
    rx = width / 2.0
    ry = height / 2.0

    # bounding box of affected pixels (with 1px margin for the anti-aliased rim)
    x0 = int(math.floor(left)) - 1
    y0 = int(math.floor(top)) - 1
    x1 = int(math.ceil(left + width)) + 1
    y1 = int(math.ceil(top + height)) + 1
    h, w = img.shape[:2]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return

    # sample the sub-pixels in the affected region
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sub_offsets = (np.arange(_SS) + 0.5) / _SS
    inside = np.zeros((y1 - y0, x1 - x0), np.float32)
    for oy in sub_offsets:
        py = yy + oy
        for ox in sub_offsets:
            px = xx + ox
            inside += (((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2) <= 1.0
    coverage = inside / (_SS * _SS)

    _blend(img, y0, y1, x0, x1, coverage, color)


def _fill_ellipse(img, left, top, width, height, color):
    """Fill an ellipse with NO anti-aliasing (GDI+ ``SmoothingMode.None``):
    a pixel is filled if its centre lies inside the ellipse."""
    if width <= 0 or height <= 0:
        return

    cx = left + width / 2.0
    cy = top + height / 2.0
    rx = width / 2.0
    ry = height / 2.0

    x0 = int(math.floor(left))
    y0 = int(math.floor(top))
    x1 = int(math.ceil(left + width))
    y1 = int(math.ceil(top + height))
    h, w = img.shape[:2]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return

    # pixel centres
    yy, xx = np.mgrid[y0:y1, x0:x1]
    py = yy + 0.5
    px = xx + 0.5
    coverage = (((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2) <= 1.0
    coverage = coverage.astype(np.float32)

    _blend(img, y0, y1, x0, x1, coverage, color)


def _blend(img, y0, y1, x0, x1, coverage, color):
    """Blend ``color`` over ``img[y0:y1, x0:x1]`` with the given coverage map
    (0..1). Handles both single-channel (scalar color) and 3-channel (RGB).

    If ``img`` is an integer array the result is cast back to ``np.uint8``;
    if it is float, the blended (float) values are kept as-is."""
    region = img[y0:y1, x0:x1].astype(np.float32)
    if region.ndim == 3:
        cov = coverage[..., None]
        col = np.array(color, np.float32).reshape(1, 1, 3)
    else:
        cov = coverage
        col = np.float32(color)
    blended = (1 - cov) * region + cov * col
    if np.issubdtype(img.dtype, np.integer):
        blended = blended.astype(np.uint8)
    img[y0:y1, x0:x1] = blended


# ---------------------------------------------------------------------------
# Text drawing (rotated, matching GDI+ RotateTransform(315))
# ---------------------------------------------------------------------------


def _draw_rotated_text(img, text, center, angle_deg, font):
    """Draw white *text* centred at *center* rotated by *angle_deg*.

    ``angle_deg`` follows OpenCV convention (positive = counter-clockwise in
    image coordinates, Y down). This matches GDI+'s ``RotateTransform``: a
    GDI+ clockwise rotation of ``a`` degrees equals an OpenCV rotation of
    ``-a`` degrees.

    ``font`` is a Pillow ``ImageFont.FreeTypeFont`` (loaded with a specific
    point size), mirroring GDI+'s ``DrawString`` with a point-sized font.
    """
    # measure the text with the exact font (GDI+ MeasureString equivalent)
    text_img = Image.new("L", (1, 1), 0)
    tmp = ImageDraw.Draw(text_img)
    tw, th = tmp.textsize(text, font=font)

    # render white text on a transparent tile with some padding for rotation
    pad = int(max(tw, th) * 1.5)
    tile = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    # Centre the rendered glyph on the tile centre. GDI+'s DrawString centres
    # the *line box* (ascent+descent+leading) on the origin, which leaves the
    # glyph slightly ABOVE the origin because descent/leading sit below it.
    # Pillow's anchor="mm" centres on the ascent+descent middle instead, so it
    # lands the glyph slightly BELOW the origin. Shift the glyph up to reproduce
    # GDI+'s placement.
    #
    # Why 0.11 * em size: the glyph's top sits ~(ascent - cap_height) below the
    # line-box top. Centring the line box (GDI+) vs. centring the ascent+descent
    # span (Pillow "mm") differs by roughly
    #     (ascent - cap_height - descent) / 2
    # For the bundled TeX Gyre Adventor at 96 px this is ~10.5 px, i.e. ~0.11 em;
    # it scales with the em size, so smaller fonts shift proportionally less.
    # The constant was calibrated so the rendered "0" and "HD23" glyph centres
    # land within ~1.5 px of the reference markers.
    vshift = font.size * 0.11
    d.text((pad, pad - vshift), text, font=font, fill=(255, 255, 255, 255), anchor="mm")

    # rotate counter-clockwise by angle_deg (OpenCV convention, positive =
    # counter-clockwise). Pillow's rotate() is also counter-clockwise for
    # positive angles, so use it directly.
    rotated = tile.rotate(angle_deg, resample=Image.BICUBIC, center=(pad, pad))

    # composite onto the numpy image at the requested centre
    arr = np.asarray(rotated).astype(np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    fg = arr[:, :, :3]

    x0 = int(round(center[0] - pad))
    y0 = int(round(center[1] - pad))
    h, w = img.shape[:2]
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    paste_w = min(pad * 2 - src_x0, w - dst_x0)
    paste_h = min(pad * 2 - src_y0, h - dst_y0)
    if paste_w <= 0 or paste_h <= 0:
        return img

    roi = img[dst_y0 : dst_y0 + paste_h, dst_x0 : dst_x0 + paste_w].astype(np.float32)
    a = alpha[src_y0 : src_y0 + paste_h, src_x0 : src_x0 + paste_w]
    f = fg[src_y0 : src_y0 + paste_h, src_x0 : src_x0 + paste_w]
    blended = (1 - a) * roi + a * f
    img[dst_y0 : dst_y0 + paste_h, dst_x0 : dst_x0 + paste_w] = blended.astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Marker rendering
# ---------------------------------------------------------------------------

# Candidate font files, tried in order. The bundled TeX Gyre Adventor is a
# free re-issue of URW Gothic (the Avant Garde Gothic clone, i.e. the same
# design Century Gothic is based on), so it is the preferred substitute for
# the original "Century Gothic". It is shipped next to this script so the
# generator is self-contained; a user-provided genuine "Century Gothic" file
# placed in the same directory takes precedence. If neither is available, the
# last-resort Pillow bitmap font is used.
_FONT_CANDIDATES = [
    "Century Gothic.ttf",
    "century gothic.ttf",
    "TexGyreAdventor-Regular.otf",
]


def _load_font(point_size):
    """Load a TrueType font at *point_size* (points, as in GDI+'s
    ``new Font(family, emSize)``), falling back through a list of candidates
    and finally to Pillow's built-in bitmap font.

    GDI+ renders at 96 DPI by default (Display unit = 1/96 inch), so a font
    size in points is converted to pixels by ``point * 96 / 72 = point * 4/3``.
    """
    pixel_size = int(round(point_size * 96.0 / 72.0))
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, pixel_size)
            except Exception:
                continue
    # last resort: default PIL font (fixed, small) — size is approximate
    return ImageFont.load_default()


def create_index_string(index, code_count):
    """Zero-padded index string, same width as the total code count."""
    return str(index).zfill(len(str(code_count)))


def read_code_list(hd):
    """Read the 48-bit codes for family *hd* from ``HD{hd}.txt``."""
    codes = []
    with open(f"HD{hd}.txt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            codes.append([int(ch) for ch in line[:NO_OF_BITS]])
    return codes


def draw_markers(hd, code_locs, nearby_codes):
    """Render all markers of family *hd* and save them under ``HD{hd}_generated/``."""
    dir_name = f"HD{hd}_generated"
    os.makedirs(dir_name, exist_ok=True)

    codes = read_code_list(hd)
    code_count = len(codes)

    # Load the text fonts. The original uses "Century Gothic" at 72 pt
    # (large, index) and 20 pt (small, "HDxx"). We look for a local font file
    # first, then fall back to any available system font of the same family.
    large_font = _load_font(72)
    small_font = _load_font(20)

    # geometry (pixels)
    marker_size = FILE_SIZE / (1 + BORDER * 2)
    border_size = marker_size * BORDER

    outer_diameter = 2 * marker_size * OUTER_CIRCLE_RADIUS
    inner_diameter = 2 * marker_size * INNER_CIRCLE_RADIUS
    outer_topleft = (FILE_SIZE - outer_diameter) / 2
    inner_topleft = (FILE_SIZE - inner_diameter) / 2
    code_diameter = 2 * inner_diameter * CODE_RADIUS
    filler_diameter = code_diameter * FILLER_CODE_RADIUS

    # precompute code-circle centre coordinates in pixels (float, matching
    # GDI+'s FillEllipse which takes floating-point geometry)
    code_centers = [
        (inner_topleft + inner_diameter * x, inner_topleft + inner_diameter * y)
        for x, y in code_locs
    ]
    # top-left corner of a code circle's bounding box (GDI+ FillEllipse takes
    # the bounding-box origin, not the centre)
    code_left = code_diameter / 2.0
    filler_left = filler_diameter / 2.0

    for i, code in enumerate(tqdm(codes, desc=f"HD{hd}", unit="marker", leave=False)):
        img = np.full((FILE_SIZE, FILE_SIZE, 3), 255, np.uint8)

        # outer rectangle: solid black, sharp (non-antialiased) corners, same
        # as GDI+ FillRectangle with SmoothingMode.None. Filled as a plain
        # numpy slice so no rounded/anti-aliased corners are introduced.
        x0 = int(round(border_size))
        y0 = int(round(border_size))
        x1 = int(round(border_size + marker_size))
        y1 = int(round(border_size + marker_size))
        img[y0:y1, x0:x1] = 0

        # outer circle (white), anti-aliased (GDI+ SmoothingMode.AntiAlias)
        _fill_ellipse_aa(
            img,
            outer_topleft,
            outer_topleft,
            outer_diameter,
            outer_diameter,
            (255, 255, 255),
        )

        # code circles (black for bit 1), no anti-aliasing (SmoothingMode.None)
        for j in range(NO_OF_BITS):
            if code[j] == 1:
                _fill_ellipse(
                    img,
                    code_centers[j][0] - code_left,
                    code_centers[j][1] - code_left,
                    code_diameter,
                    code_diameter,
                    (0, 0, 0),
                )

        # filler circles between nearby pairs that are both 1 (no AA)
        for j in range(NO_OF_BITS):
            if code[j] != 1:
                continue
            for k in nearby_codes[j]:
                if code[k] == 1 and k > j:
                    cx = (code_centers[j][0] + code_centers[k][0]) / 2.0
                    cy = (code_centers[j][1] + code_centers[k][1]) / 2.0
                    _fill_ellipse(
                        img,
                        cx - filler_left,
                        cy - filler_left,
                        filler_diameter,
                        filler_diameter,
                        (0, 0, 0),
                    )

        # white code circles (bit 0), no anti-aliasing
        for j in range(NO_OF_BITS):
            if code[j] == 0:
                _fill_ellipse(
                    img,
                    code_centers[j][0] - code_left,
                    code_centers[j][1] - code_left,
                    code_diameter,
                    code_diameter,
                    (255, 255, 255),
                )

        # morphological close (dilate then erode) repeated
        border_px = int(outer_topleft)
        for _ in range(MORPH_ITERATIONS):
            img = _dilate(img, MORPH_RADIUS, border_px)
            img = _erode(img, MORPH_RADIUS, border_px)

        # smooth, then re-draw the code circles with high quality (anti-aliased)
        img = _smooth(img, border_px)

        for j in range(NO_OF_BITS):
            color = (0, 0, 0) if code[j] == 1 else (255, 255, 255)
            _fill_ellipse_aa(
                img,
                code_centers[j][0] - code_left,
                code_centers[j][1] - code_left,
                code_diameter,
                code_diameter,
                color,
            )

        # Fill the ring between the inner and outer circles with WHITE.
        # The original draws a black bitmap, fills the outer circle white
        # (HighQuality AA) and the inner circle black (None), then
        # MakeTransparent(Color.Black) keeps only the white ring, which is
        # drawn over the marker. Net effect: the annulus becomes white with an
        # anti-aliased outer edge and a hard inner edge.
        #
        # IMPORTANT: the outer circle is anti-aliased, so its rim has grey
        # pixels (partial coverage). These must be alpha-blended onto the image
        # (as GDI+'s DrawImage does), NOT thresholded: `img[ring > 0] = 255`
        # would flatten every grey rim pixel to solid white and lose the AA.
        ring = np.zeros((FILE_SIZE, FILE_SIZE), np.float32)
        _fill_ellipse_aa(
            ring, outer_topleft, outer_topleft, outer_diameter, outer_diameter, 1.0
        )
        _fill_ellipse(
            ring,
            inner_topleft - 1,
            inner_topleft - 1,
            inner_diameter + 2,
            inner_diameter + 2,
            0.0,
        )
        # blend white over the image using the ring's coverage (grey = partial)
        img = (
            (1.0 - ring[..., None]) * img.astype(np.float32) + ring[..., None] * 255.0
        ).astype(np.uint8)

        # strings: large index + small "HDxx", both white. GDI+'s
        # RotateTransform(315) is a 315 deg CLOCKWISE rotation (GDI+ Y is down,
        # positive = clockwise), which equals 45 deg counter-clockwise here.
        index_str = create_index_string(i, code_count)
        c1 = 2.375 * border_size
        _draw_rotated_text(img, index_str, (c1, c1), 45, large_font)

        hd_str = f"HD{hd}"
        c2 = 1.875 * border_size
        _draw_rotated_text(img, hd_str, (c2, c2), 45, small_font)

        id_str = index_str.zfill(5)
        cv2.imwrite(os.path.join(dir_name, f"{id_str}.png"), img)


# all supported Hamming-distance families (odd HD from 11 to 23)
ALL_HD = list(range(11, 24, 2))


def _parse_hd_arg(value):
    """Parse the ``--HD`` option into a list of HD families to generate.

    ``value`` may be:
      - ``"all"``            -> every supported family (ALL_HD)
      - a single integer     -> just that family, e.g. ``23``
      - comma-separated ints -> several families, e.g. ``11,15,23``

    Raises ``SystemExit`` on an unparseable value.
    """
    value = value.strip()
    if value.lower() == "all":
        return ALL_HD

    hd_list = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            hd_list.append(int(part))
        except ValueError:
            raise SystemExit(
                f"invalid --HD value {part!r}: expected 'all' or an integer"
            )
    if not hd_list:
        raise SystemExit("invalid --HD value: empty")
    return hd_list


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate STag markers for the requested Hamming-distance families."
    )
    parser.add_argument(
        "--HD",
        default="all",
        help="which HD family/families to generate: 'all', a single value "
        "(e.g. 23), or comma-separated values (e.g. 11,15,23). Default: all.",
    )
    args = parser.parse_args()

    hd_list = _parse_hd_arg(args.HD)

    code_locs, nearby_codes = fill_locs()
    for hd in hd_list:
        draw_markers(hd, code_locs, nearby_codes)
        print(f"Generated markers for HD{hd}")


if __name__ == "__main__":
    main()
