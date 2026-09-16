#!/usr/bin/env python3
"""
hand_frame_fx.py — Hold both hands up like a director framing a shot.
The four fingertips (index tip + thumb tip of each hand) define a free-form
quadrilateral. A stylized effect is rendered ONLY inside that shape,
composited live over the webcam feed.

Controls:
    r      start / stop recording (.mp4, saved next to this script)
    space  advance to the next effect manually
    l      toggle hand-skeleton overlay (debug)
    q      quit (finalizes / saves any active recording)

Hands-free switching: bring your two hands close together (like clapping)
to advance to the next effect. Useful while recording so you never have
to touch the keyboard.

Run:
    pip install -r requirements.txt
    python3 hand_frame_fx.py
"""

import os
import sys
import time
import argparse
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# --------------------------------------------------------------------------
# Effects — each takes a BGR patch (H x W x 3 uint8) and returns a same-size
# BGR patch. Kept vectorized / cheap since they run on every frame.
# --------------------------------------------------------------------------

def comic(patch):
    """Pop-art posterize: grayscale -> equalizeHist -> 4-level posterize ->
    red -> orange -> yellow -> white gradient colormap."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    levels = np.clip((eq.astype(np.int32) * 4) // 256, 0, 3)
    palette = np.array([
        [40, 40, 160],    # dark red   (BGR)
        [0, 130, 255],    # orange
        [0, 225, 255],    # yellow
        [255, 255, 255],  # white
    ], dtype=np.uint8)
    return palette[levels]


def paper(patch):
    """Black ink outline over dotted stipple shading on warm cream paper."""
    h, w = patch.shape[:2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    cream = np.full((h, w, 3), (225, 240, 247), dtype=np.uint8)  # BGR cream

    # Halftone stipple: darker regions get bigger dots.
    cell = max(6, min(h, w) // 30)
    small_w = max(1, w // cell)
    small_h = max(1, h // cell)
    dark = 255.0 - cv2.resize(gray, (small_w, small_h),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
    dark /= 255.0

    stipple = cream.copy()
    for j in range(small_h):
        cy = j * cell + cell // 2
        for i in range(small_w):
            r = dark[j, i]
            if r > 0.10:
                radius = max(1, int(r * cell * 0.55))
                cx = i * cell + cell // 2
                cv2.circle(stipple, (cx, cy), radius, (172, 176, 180), -1, cv2.LINE_AA)

    # Ink outline via adaptive threshold.
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 6)
    ink_mask = edges < 128
    stipple[ink_mask] = (35, 30, 28)
    return stipple


def grid(patch):
    """Greyscale subject under a crisp technical grid (major lines every 4th)."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    step = max(6, min(h, w) // 24)
    minor_color = (95, 95, 95)
    major_color = (60, 220, 255)  # bright amber accent (BGR)

    for x in range(0, w, step):
        idx = x // step
        major = (idx % 4 == 0)
        cv2.line(base, (x, 0), (x, h),
                  major_color if major else minor_color, 2 if major else 1, cv2.LINE_AA)
    for y in range(0, h, step):
        idx = y // step
        major = (idx % 4 == 0)
        cv2.line(base, (0, y), (w, y),
                  major_color if major else minor_color, 2 if major else 1, cv2.LINE_AA)
    return base


def pixel_glass(patch):
    """Vibrant frosted base broken into chunky glass tiles, each with a
    bright highlight corner."""
    h, w = patch.shape[:2]
    blurred = cv2.GaussianBlur(patch, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)   # saturation boost
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.05, 0, 255)  # slight brighten
    vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    tile = max(10, min(h, w) // 18)
    small = cv2.resize(vibrant, (max(1, w // tile), max(1, h // tile)),
                        interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    out = mosaic.copy()
    for ty in range(0, h, tile):
        th = min(tile, h - ty)
        hs = min(max(2, tile // 4), th)
        for tx in range(0, w, tile):
            tw = min(tile, w - tx)
            ws = min(max(2, tile // 4), tw)
            roi = out[ty:ty + hs, tx:tx + ws]
            roi[:] = np.clip(roi.astype(np.int32) + 70, 0, 255).astype(np.uint8)
    return out


EFFECTS = [comic, paper, grid, pixel_glass]
EFFECT_NAMES = ["comic", "paper", "grid", "pixel glass"]


def run_at_working_res(patch, fn, max_dim=420):
    """Downscale patch to a capped working resolution, run the effect, then
    upscale back to the original patch size so a large on-screen quad can't
    tank FPS."""
    h, w = patch.shape[:2]
    if h == 0 or w == 0:
        return patch
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(patch, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
        processed = fn(small)
        return cv2.resize(processed, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn(patch)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class HandFrameApp:
    def __init__(self, camera_index=0):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Shape smoothing / anti-flicker state.
        self.ema_points = None          # 4 points: [thumb_L, index_L, thumb_R, index_R]
        self.ema_alpha = 0.55
        self.dropout_count = 0
        self.dropout_max = 5

        # Hands-free filter switching state.
        self.armed = True
        self.switch_cooldown = 0
        self.trigger_ratio = 1.5
        self.rearm_ratio = 2.6

        self.filter_idx = 0
        self.show_skeleton = False

        # Recording state.
        self.writer = None
        self.record_start_time = None
        self.frames_written = 0
        self.record_path = None
        self.record_fps = 30
        self.max_burst = 15

        self.fps_hist = deque(maxlen=20)
        self.last_time = time.time()

    # ------------------------------------------------------------------
    def landmark_xy(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def extract_hands(self, results, w, h):
        """Returns list of dicts with thumb, index, wrist, index_mcp, center, span
        in full-resolution pixel coordinates."""
        out = []
        if not results.multi_hand_landmarks:
            return out
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            thumb = pts[4]
            index = pts[8]
            wrist = pts[0]
            index_mcp = pts[5]
            center = pts.mean(axis=0)
            span = max(1.0, dist(index_mcp, wrist))
            out.append({
                "thumb": thumb, "index": index, "wrist": wrist,
                "index_mcp": index_mcp, "center": center, "span": span,
                "landmarks": hand_landmarks,
            })
        return out

    # ------------------------------------------------------------------
    def update_shape(self, hands_info):
        """Update the EMA'd 4-corner quad. Returns hull points (Nx2 float32)
        or None if no valid shape is currently available."""
        if len(hands_info) == 2:
            hands_sorted = sorted(hands_info, key=lambda hnd: hnd["center"][0])
            left, right = hands_sorted[0], hands_sorted[1]
            raw = np.array([left["thumb"], left["index"],
                             right["thumb"], right["index"]], dtype=np.float32)

            if self.ema_points is None:
                self.ema_points = raw.copy()
            else:
                self.ema_points = (self.ema_alpha * self.ema_points +
                                    (1 - self.ema_alpha) * raw)
            self.dropout_count = 0

            # Hands-free filter switching.
            avg_span = (left["span"] + right["span"]) / 2.0
            centers_dist = dist(left["center"], right["center"])
            ratio = centers_dist / max(1.0, avg_span)

            if self.switch_cooldown > 0:
                self.switch_cooldown -= 1
            else:
                if self.armed and ratio < self.trigger_ratio:
                    self.advance_filter()
                    self.armed = False
                    self.switch_cooldown = 5
                elif not self.armed and ratio > self.rearm_ratio:
                    self.armed = True
        else:
            # No valid 2-hand detection this frame -> anti-flicker hold.
            if self.ema_points is not None and self.dropout_count < self.dropout_max:
                self.dropout_count += 1
            else:
                self.ema_points = None

        if self.ema_points is None:
            return None

        hull = cv2.convexHull(self.ema_points.astype(np.float32))
        hull = hull.reshape(-1, 2)
        if len(hull) < 3:
            return None
        return hull

    def advance_filter(self):
        self.filter_idx = (self.filter_idx + 1) % len(EFFECTS)

    # ------------------------------------------------------------------
    def composite_effect(self, frame, hull):
        h, w = frame.shape[:2]
        xs, ys = hull[:, 0], hull[:, 1]
        x0 = int(max(0, np.floor(xs.min())))
        y0 = int(max(0, np.floor(ys.min())))
        x1 = int(min(w, np.ceil(xs.max())))
        y1 = int(min(h, np.ceil(ys.max())))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return

        patch = frame[y0:y1, x0:x1].copy()
        effect_fn = EFFECTS[self.filter_idx]
        effect_patch = run_at_working_res(patch, effect_fn, max_dim=420)

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = (hull - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(mask, shifted, 255)
        mask_bool = mask.astype(bool)

        roi = frame[y0:y1, x0:x1]
        roi[mask_bool] = effect_patch[mask_bool]
        frame[y0:y1, x0:x1] = roi

    # ------------------------------------------------------------------
    def start_recording(self, w, h):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(script_dir, f"hand_frame_fx_{timestamp}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = base + ".mp4"
        writer = cv2.VideoWriter(path, fourcc, self.record_fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            path = base + ".avi"
            writer = cv2.VideoWriter(path, fourcc, self.record_fps, (w, h))

        if not writer.isOpened():
            print("ERROR: could not open a video writer (mp4v or XVID).")
            return

        self.writer = writer
        self.record_path = path
        self.record_start_time = time.time()
        self.frames_written = 0
        print(f"Recording started -> {path}")

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Recording saved -> {self.record_path}")
        self.writer = None
        self.record_start_time = None
        self.frames_written = 0

    def record_tick(self, clean_frame):
        """Write frames on a wall-clock schedule so playback stays real-time
        even if processing runs slower than the declared fps."""
        if self.writer is None:
            return
        elapsed = time.time() - self.record_start_time
        target_frames = int(elapsed * self.record_fps)
        to_write = target_frames - self.frames_written
        if to_write <= 0:
            return
        to_write = min(to_write, self.max_burst)
        for _ in range(to_write):
            self.writer.write(clean_frame)
            self.frames_written += 1

    # ------------------------------------------------------------------
    def run(self):
        if not self.cap.isOpened():
            print("ERROR: could not open webcam.")
            return

        print("hand_frame_fx running. Keys: [r]ecord  [space] next effect  "
              "[l] skeleton  [q] quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            small = cv2.resize(frame, (w // 2, h // 2))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)

            hands_info = self.extract_hands(results, w, h)
            hull = self.update_shape(hands_info)
            if hull is not None:
                self.composite_effect(frame, hull)

            # Clean composited frame captured BEFORE any overlays, so
            # recordings never contain the FPS counter / REC badge / skeleton.
            clean_frame = frame.copy()
            self.record_tick(clean_frame)

            # ---- overlays (display only, never recorded) ----
            if self.show_skeleton and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                self.fps_hist.append(1.0 / dt)
            fps = sum(self.fps_hist) / len(self.fps_hist) if self.fps_hist else 0.0

            cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 150, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 150, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, EFFECT_NAMES[self.filter_idx], (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, EFFECT_NAMES[self.filter_idx], (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if self.writer is not None:
                cv2.circle(frame, (w - 170, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, "REC", (w - 155, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("hand_frame_fx", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                if self.writer is None:
                    self.start_recording(w, h)
                else:
                    self.stop_recording()
            elif key == ord(" "):
                self.advance_filter()
            elif key == ord("l"):
                self.show_skeleton = not self.show_skeleton

        self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    parser = argparse.ArgumentParser(description="Hand-frame FX webcam app")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()
    app = HandFrameApp(camera_index=args.camera)
    app.run()


if __name__ == "__main__":
    main()
