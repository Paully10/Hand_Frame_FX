#!/usr/bin/env python3
"""
make_contact_sheet.py — Applies all 4 hand_frame_fx effects to a source
image and tiles them (original + each effect, labeled) into
effects_contact_sheet.png.

Usage:
    python3 make_contact_sheet.py                 # synthetic demo scene
    python3 make_contact_sheet.py cam              # grab one webcam frame
    python3 make_contact_sheet.py path/to/image.jpg
"""

import os
import sys

import cv2
import numpy as np

from hand_frame_fx import comic, paper, grid, pixel_glass, run_at_working_res


def make_demo_scene(w=640, h=480):
    """A synthetic, tonally-rich scene: gradient sky, shapes with shading,
    and text — enough contrast/edges/color range to show off all 4 effects."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Diagonal gradient background.
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grad = (0.5 * gx + 0.5 * gy)
    top_color = np.array([70, 40, 20], dtype=np.float32)    # deep blue (BGR)
    bot_color = np.array([230, 200, 140], dtype=np.float32)  # warm light
    for c in range(3):
        img[..., c] = (top_color[c] * (1 - grad) + bot_color[c] * grad).astype(np.uint8)

    # Bright sun / circle with radial shading.
    cv2.circle(img, (int(w * 0.75), int(h * 0.25)), 60, (60, 210, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(w * 0.75), int(h * 0.25)), 60, (30, 150, 220), 3, cv2.LINE_AA)

    # Mountains (dark shapes, high contrast edges).
    pts1 = np.array([[0, h], [w * 0.15, h * 0.55], [w * 0.35, h * 0.85],
                      [w * 0.5, h * 0.6], [w * 0.65, h], ], dtype=np.int32)
    cv2.fillPoly(img, [pts1], (40, 60, 40))
    pts2 = np.array([[w * 0.4, h], [w * 0.62, h * 0.5], [w * 0.8, h * 0.8],
                      [w * 1.0, h * 0.62], [w, h]], dtype=np.int32)
    cv2.fillPoly(img, [pts2], (25, 40, 25))

    # Foreground rectangle "building" with window grid (good for edges/grid).
    cv2.rectangle(img, (int(w * 0.08), int(h * 0.62)), (int(w * 0.28), int(h * 0.95)),
                  (60, 60, 90), -1)
    for wy in range(int(h * 0.66), int(h * 0.92), 18):
        for wx in range(int(w * 0.11), int(w * 0.26), 16):
            cv2.rectangle(img, (wx, wy), (wx + 8, wy + 10), (150, 210, 250), -1)

    # A few saturated shapes for the pixel-glass / comic colormaps.
    cv2.circle(img, (int(w * 0.5), int(h * 0.78)), 40, (180, 60, 200), -1, cv2.LINE_AA)
    cv2.rectangle(img, (int(w * 0.55), int(h * 0.65)), (int(w * 0.65), int(h * 0.85)),
                  (60, 180, 90), -1)

    cv2.putText(img, "HAND FRAME FX", (int(w * 0.06), int(h * 0.15)),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "HAND FRAME FX", (int(w * 0.06), int(h * 0.15)),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 20), 1, cv2.LINE_AA)

    return img


def load_source(arg):
    if arg is None:
        print("Using synthetic demo scene.")
        return make_demo_scene()
    if arg == "cam":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        ok, frame = False, None
        for _ in range(10):  # warm up a few frames
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("ERROR: could not grab a webcam frame; falling back to demo scene.")
            return make_demo_scene()
        return cv2.flip(frame, 1)
    if os.path.isfile(arg):
        img = cv2.imread(arg)
        if img is None:
            print(f"ERROR: could not read image at {arg}; falling back to demo scene.")
            return make_demo_scene()
        return img
    print(f"ERROR: '{arg}' not found; falling back to demo scene.")
    return make_demo_scene()


def make_tile(img, label, tile_size=340, label_h=34):
    resized = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_size + label_h, tile_size, 3), dtype=np.uint8)
    canvas[:tile_size, :, :] = resized
    canvas[tile_size:, :, :] = (25, 25, 25)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    tx = (tile_size - text_size[0]) // 2
    ty = tile_size + label_h - 10
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_contact_sheet(source_img, out_path="effects_contact_sheet.png"):
    tiles_data = [
        ("Original", source_img),
        ("Comic", run_at_working_res(source_img, comic)),
        ("Paper", run_at_working_res(source_img, paper)),
        ("Grid", run_at_working_res(source_img, grid)),
        ("Pixel Glass", run_at_working_res(source_img, pixel_glass)),
    ]

    tile_size = 340
    label_h = 34
    pad = 14
    cols = 3
    rows = 2

    tiles = [make_tile(img, label, tile_size, label_h) for label, img in tiles_data]
    tile_h = tile_size + label_h

    sheet_w = cols * tile_size + (cols + 1) * pad
    sheet_h = rows * tile_h + (rows + 1) * pad + 50
    sheet = np.full((sheet_h, sheet_w, 3), (18, 18, 18), dtype=np.uint8)

    cv2.putText(sheet, "hand_frame_fx — effects contact sheet", (pad, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    y_offset = 50
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_size + pad)
        y = y_offset + pad + r * (tile_h + pad)
        sheet[y:y + tile_h, x:x + tile_size] = tile

    cv2.imwrite(out_path, sheet)
    print(f"Contact sheet saved -> {out_path}")
    return out_path


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    source_img = load_source(arg)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "effects_contact_sheet.png")
    build_contact_sheet(source_img, out_path)


if __name__ == "__main__":
    main()
