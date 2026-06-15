"""
Real-Time YOLOv8 Camera Object-Trajectory Segmentation
────────────────────────────────────────────────────────
Merged: Code 1 base + Code 2 stabilized tracking for pick-and-pour / bending

Key changes vs original Code 1
───────────────────────────────
1.  best_detection()  →  now picks the detection NEAREST to last anchor
    (Code 2 strategy). Prevents bbox jumping to a different high-confidence
    detection while the bottle is tilting.

2.  OpenCV CSRT/KCF tracker fallback (from Code 2).
    When YOLO misses (blur, occlusion during pour), the tracker continues
    the trajectory smoothly instead of holding a stale position.

3.  Smoothed bbox  (new).
    An EMA is applied to the raw bbox coordinates before the anchor point
    is computed. This prevents the anchor from snapping when the bbox
    reshapes during tilt/pour.

4.  Everything else in Code 1 is preserved exactly:
    graph style, CSV outputs, keyboard controls, segmentation logic,
    live-graph window, start-detection, EMA+Savgol smoothing on points.
"""

print("SCRIPT STARTED", flush=True)

import os
import csv
import json
import math
import platform
from datetime import datetime

import cv2
import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  ── edit only this section
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX       = 0
TARGET_CLASS       = "bottle"
CONF_THRESHOLD     = 0.8
YOLO_IMGSZ         = 450

UPPER_ANCHOR_RATIO = 0.22       # fraction of bbox height from top

SHOW_WINDOWS           = True
SHOW_CURRENT_POINT     = True
SAVE_ANNOTATED_VIDEO   = True
SHOW_LIVE_GRAPH_WINDOW = True

PLAYBACK_DELAY_MS = 1

# ── NEW: OpenCV tracker fallback settings (from Code 2) ──────────────────────
USE_OPENCV_TRACKER_WHEN_YOLO_MISSES = True
MAX_TRACKER_ONLY_FRAMES             = 45   # give up after this many consecutive tracker frames

# ── NEW: Bbox smoothing (prevents anchor jump during tilt/reshape) ────────────
BBOX_EMA_ALPHA = 0.45   # lower = smoother bbox, higher = more responsive

# Output root
if os.path.exists("D:\\"):
    BASE_OUTPUT_DIR = r"D:\trajectory_output"
else:
    BASE_OUTPUT_DIR = os.path.join(
        os.path.expanduser("~"), "Desktop", "trajectory_output"
    )

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
CURRENT_DIR = os.getcwd()

MODEL_PATH_CANDIDATES = [
    os.path.join(CURRENT_DIR, "yolo-Weights", "yolov8n.pt"),
    os.path.join(SCRIPT_DIR,  "yolo-Weights", "yolov8n.pt"),
    r"C:\Users\srish\OneDrive\Documents\yolo\yolo-Weights\yolov8n.pt",
    "yolov8n.pt",
]

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

YOLO_EVERY_N_FRAMES = 1
MAX_HOLD_FRAMES     = 5

SAVGOL_WINDOW    = 11
SAVGOL_POLYORDER = 2
EMA_ALPHA        = 0.35

START_SPEED_PIXELS_PER_FRAME      = 0.35
START_MOTION_CONSECUTIVE_FRAMES   = 3
START_MOTION_DISTANCE_PIXELS      = 4.0
START_REST_BACKTRACK_FRAMES       = 1
FORCE_START_AFTER_VALID_POINTS    = 5

MIN_BEND_ANGLE_DEG              = 30.0
BEND_WINDOW_FRAMES              = 5
MIN_BOUNDARY_GAP_FRAMES         = 10
FIRST_BOUNDARY_MIN_GAP_FRAMES   = 3
MIN_SEGMENT_DISPLACEMENT_PIXELS = 6.0
MIN_SEGMENT_LENGTH_FRAMES       = 5
MAX_BOUNDARIES                  = None

FIGURE_SIZE          = (10, 7)
FIGURE_DPI           = 220
SAVE_DPI             = 350
GRAPH_PADDING_PIXELS = 100
MARKER_SIZE          = 90

X_LABEL_TEXT = "X"
Y_LABEL_TEXT = "Y"
FOOTER_TEXT  = ""

AXIS_LABEL_FONTSIZE   = 13
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE   = 11
LEGEND_FONTSIZE       = 11
LEGEND_TITLE_FONTSIZE = 12

LIVE_GRAPH_UPDATE_EVERY_N_FRAMES = 5
LIVE_GRAPH_WIN_W = 900
LIVE_GRAPH_WIN_H = 620
LIVE_GRAPH_DPI   = 130


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATH MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

RUN_NAME     = "camera_realtime_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
os.makedirs(RUN_ROOT_DIR, exist_ok=True)

master_summary_csv = os.path.join(BASE_OUTPUT_DIR, "all_trajectories_summary.csv")

SAVE_AND_NEXT_KEY      = ord("s")
CURRENT_TRAJECTORY_ID  = 1
ACTIVE_CAMERA_INDEX    = CAMERA_INDEX

OUTPUT_DIR                   = ""
output_video_path            = ""
framewise_csv_path           = ""
boundary_csv_path            = ""
segment_csv_path             = ""
trajectory_summary_csv_path  = ""
json_path                    = ""
final_image_path             = ""
preview_image_path           = ""


def set_trajectory_output_paths(trajectory_id):
    global CURRENT_TRAJECTORY_ID, OUTPUT_DIR
    global output_video_path, framewise_csv_path
    global boundary_csv_path, segment_csv_path
    global trajectory_summary_csv_path, json_path
    global final_image_path, preview_image_path

    CURRENT_TRAJECTORY_ID = int(trajectory_id)
    prefix = f"trajectory_{CURRENT_TRAJECTORY_ID:03d}"

    OUTPUT_DIR = os.path.join(RUN_ROOT_DIR, prefix)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_video_path           = os.path.join(OUTPUT_DIR, f"{prefix}_annotated_video.mp4")
    framewise_csv_path          = os.path.join(OUTPUT_DIR, "framewise_data.csv")
    boundary_csv_path           = os.path.join(OUTPUT_DIR, "boundary_points.csv")
    segment_csv_path            = os.path.join(OUTPUT_DIR, "temporal_segments.csv")
    trajectory_summary_csv_path = os.path.join(OUTPUT_DIR, "trajectory_summary.csv")
    json_path                   = os.path.join(OUTPUT_DIR, "segmentation_output.json")
    final_image_path            = os.path.join(OUTPUT_DIR, f"{prefix}_trajectory_graph.png")
    preview_image_path          = os.path.join(OUTPUT_DIR, f"{prefix}_trajectory_graph_preview.png")


set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def segment_label(seg_id):
    return f"Segment_{seg_id}"


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_finite(p):
    if p is None:
        return False
    try:
        return np.isfinite(float(p[0])) and np.isfinite(float(p[1]))
    except Exception:
        return False


def is_nan(v):
    try:
        return bool(np.isnan(float(v)))
    except Exception:
        return True


def clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 2)
    if y2 <= y1: y2 = min(h - 1, y1 + 2)
    return x1, y1, x2, y2


def bbox_metrics(bbox):
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    return {"bbox_width": bw, "bbox_height": bh,
            "bbox_area": bw * bh, "aspect_ratio": bw / bh}


def upper_anchor(bbox):
    """Compute upper-anchor point from a (possibly smoothed) bbox."""
    x1, y1, x2, y2 = bbox
    bh = max(1.0, float(y2 - y1))
    return float((x1 + x2) / 2.0), float(y1 + UPPER_ANCHOR_RATIO * bh)


# ── FIX 1: nearest-to-last-anchor selection (Code 2 strategy) ────────────────
def best_detection(detections, last_anchor):
    """
    Previously: pick highest-confidence detection.
    Now: pick detection NEAREST to the last known anchor position.

    During pick-and-pour the bottle may coexist with other high-confidence
    detections (e.g. a cup, another bottle). Nearest-anchor selection keeps
    the tracker locked on the correct object through tilt and occlusion.
    If there is no prior anchor yet, fall back to highest confidence.
    """
    if not detections:
        return None
    if last_anchor is None:
        # First frame — no prior position, use confidence as tiebreaker
        return max(detections, key=lambda d: d["confidence"])
    return min(detections, key=lambda d: euclidean(d["anchor"], last_anchor))


# ── FIX 2: bbox EMA smoothing (prevents anchor snap during bbox reshape) ──────
def smooth_bbox(raw_bbox, prev_smooth_bbox, alpha):
    """
    Apply EMA to the raw bbox coordinates.
    This stabilises the anchor point when YOLO's bbox reshapes abruptly
    during tilt/pour (the bottle changes from tall-narrow to wide-short).
    alpha: higher = more responsive, lower = smoother.
    """
    if prev_smooth_bbox is None:
        return tuple(float(v) for v in raw_bbox)
    return tuple(
        alpha * float(r) + (1.0 - alpha) * float(p)
        for r, p in zip(raw_bbox, prev_smooth_bbox)
    )


# ── FIX 3: OpenCV tracker creation (Code 2 strategy) ─────────────────────────
def create_cv_tracker():
    """
    Try CSRT (most accurate) → KCF (faster) → MIL (fallback).
    Handles API differences across OpenCV versions.
    Returns None if no tracker is available.
    """
    if hasattr(cv2, "legacy"):
        for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
            if hasattr(cv2.legacy, name):
                try:
                    return getattr(cv2.legacy, name)()
                except Exception:
                    pass
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        if hasattr(cv2, name):
            try:
                return getattr(cv2, name)()
            except Exception:
                pass
    return None


def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return (x, y, x + w, y + h)


def angle_between(v1, v2):
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_v = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_v)))


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def savgol_smooth(points):
    if len(points) < 3:
        return points[-1]
    local = points[-SAVGOL_WINDOW:] if len(points) >= SAVGOL_WINDOW else points[:]
    n   = len(local)
    win = SAVGOL_WINDOW if SAVGOL_WINDOW <= n else (n if n % 2 == 1 else n - 1)
    if win % 2 == 0: win -= 1
    if win < 3:      return local[-1]
    poly = min(SAVGOL_POLYORDER, win - 1)
    xs   = np.array([p[0] for p in local], dtype=np.float32)
    ys   = np.array([p[1] for p in local], dtype=np.float32)
    try:
        sx = savgol_filter(xs, window_length=win, polyorder=poly, mode="interp")
        sy = savgol_filter(ys, window_length=win, polyorder=poly, mode="interp")
        return float(sx[-1]), float(sy[-1])
    except Exception:
        return local[-1]


def ema_filter(cur, prev, alpha):
    if prev is None:
        return cur
    return (alpha * cur[0] + (1 - alpha) * prev[0],
            alpha * cur[1] + (1 - alpha) * prev[1])


# ══════════════════════════════════════════════════════════════════════════════
# START DETECTION  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def detect_start(smooth_points, frame_numbers):
    if len(smooth_points) < START_MOTION_CONSECUTIVE_FRAMES + 3:
        return None, None
    pts    = np.array(smooth_points, dtype=np.float32)
    speeds = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    run    = 0
    fi     = None
    for i, spd in enumerate(speeds):
        if spd >= START_SPEED_PIXELS_PER_FRAME:
            run += 1
            if run >= START_MOTION_CONSECUTIVE_FRAMES:
                fi = max(0, i - START_MOTION_CONSECUTIVE_FRAMES + 1)
                break
        else:
            run = 0
    if fi is None:
        if len(smooth_points) >= FORCE_START_AFTER_VALID_POINTS:
            return 0, int(frame_numbers[0])
        return None, None
    ri   = max(0, fi - START_REST_BACKTRACK_FRAMES)
    li   = min(len(pts) - 1, ri + START_MOTION_CONSECUTIVE_FRAMES + 3)
    disp = float(np.linalg.norm(pts[li] - pts[ri]))
    if disp < START_MOTION_DISTANCE_PIXELS:
        if len(smooth_points) >= FORCE_START_AFTER_VALID_POINTS:
            return 0, int(frame_numbers[0])
        return None, None
    return ri, int(frame_numbers[ri])


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def detect_boundaries(smooth_points, frame_numbers, start_index):
    if start_index is None:
        return []
    pts_a  = smooth_points[start_index:]
    frms_a = frame_numbers[start_index:]
    if len(pts_a) < (2 * BEND_WINDOW_FRAMES + 1):
        return []
    pts_np     = np.array(pts_a, dtype=np.float32)
    candidates = []
    for i in range(BEND_WINDOW_FRAMES, len(pts_np) - BEND_WINDOW_FRAMES):
        v1 = pts_np[i]             - pts_np[i - BEND_WINDOW_FRAMES]
        v2 = pts_np[i + BEND_WINDOW_FRAMES] - pts_np[i]
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 < MIN_SEGMENT_DISPLACEMENT_PIXELS or d2 < MIN_SEGMENT_DISPLACEMENT_PIXELS:
            continue
        ang = angle_between(v1, v2)
        if ang >= MIN_BEND_ANGLE_DEG:
            candidates.append({
                "frame_number": int(frms_a[i]),
                "x": float(pts_a[i][0]),
                "y": float(pts_a[i][1]),
                "bend_angle": float(ang),
                "score":      float(ang / max(MIN_BEND_ANGLE_DEG, 1e-6)),
            })
    if not candidates:
        return []

    selected = []
    for c in candidates:
        cf = int(c["frame_number"])
        if not selected:
            if cf - int(frms_a[0]) < FIRST_BOUNDARY_MIN_GAP_FRAMES:
                continue
            selected.append(c)
        else:
            lf = int(selected[-1]["frame_number"])
            if cf - lf < MIN_BOUNDARY_GAP_FRAMES:
                if c["bend_angle"] > selected[-1]["bend_angle"]:
                    selected[-1] = c
            else:
                selected.append(c)
        if MAX_BOUNDARIES is not None and len(selected) >= MAX_BOUNDARIES:
            break

    boundaries = []
    for i, c in enumerate(selected, start=1):
        boundaries.append({
            "boundary_id":      i,
            "frame_number":     int(c["frame_number"]),
            "x":                float(c["x"]),
            "y":                float(c["y"]),
            "direction_change": float(c["bend_angle"]),
            "boundary_score":   float(c["score"]),
            "cue_type":         "bend_angle_ge_45deg",
        })
    return boundaries


def clean_boundaries(boundaries, start_frame, total_frames):
    if start_frame is None:
        return []
    cleaned    = []
    prev_start = int(start_frame)
    for b in sorted(boundaries, key=lambda x: x["frame_number"]):
        bf = int(b["frame_number"])
        if bf <= prev_start:
            continue
        if bf - prev_start + 1 < MIN_SEGMENT_LENGTH_FRAMES:
            continue
        if cleaned and bf - int(cleaned[-1]["frame_number"]) < MIN_BOUNDARY_GAP_FRAMES:
            continue
        cleaned.append(dict(b))
        prev_start = bf + 1
    while cleaned and total_frames - int(cleaned[-1]["frame_number"]) < MIN_SEGMENT_LENGTH_FRAMES:
        cleaned.pop()
    for i, b in enumerate(cleaned, start=1):
        b["boundary_id"] = i
    return cleaned


def make_segments(boundaries, start_frame, total_frames):
    if total_frames <= 0 or start_frame is None:
        return []
    cleaned = clean_boundaries(boundaries, start_frame, total_frames)
    segs    = []
    seg_st  = int(start_frame)
    for b in cleaned:
        bf = int(b["frame_number"])
        if bf < seg_st:
            continue
        sid = len(segs) + 1
        segs.append({"segment_id": sid, "label": segment_label(sid),
                     "start_frame": seg_st, "end_frame": bf,
                     "duration_frames": bf - seg_st + 1})
        seg_st = bf + 1
    if seg_st <= total_frames:
        sid = len(segs) + 1
        segs.append({"segment_id": sid, "label": segment_label(sid),
                     "start_frame": seg_st, "end_frame": total_frames,
                     "duration_frames": total_frames - seg_st + 1})
    return segs


def seg_for_frame(frame_no, boundaries, start_frame):
    if start_frame is None or frame_no < start_frame:
        return 0, "Before_START"
    sid = 1
    for b in sorted(boundaries, key=lambda x: x["frame_number"]):
        if frame_no > b["frame_number"]:
            sid += 1
        else:
            break
    return sid, segment_label(sid)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA FEED OVERLAY  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame, smooth_points, boundaries, start_point, start_frame):
    valid = [(int(p[0]), int(p[1])) for p in smooth_points if is_finite(p)]
    for i in range(1, len(valid)):
        cv2.line(frame, valid[i - 1], valid[i], (255, 0, 0), 3, cv2.LINE_AA)

    if start_point is not None and start_frame is not None:
        cv2.circle(frame, (int(start_point[0]), int(start_point[1])),
                   9, (0, 255, 0), -1, cv2.LINE_AA)

    for b in boundaries:
        cv2.circle(frame, (int(b["x"]), int(b["y"])),
                   10, (0, 0, 255), -1, cv2.LINE_AA)

    if SHOW_CURRENT_POINT and smooth_points:
        p = smooth_points[-1]
        if is_finite(p):
            cv2.circle(frame, (int(p[0]), int(p[1])),
                       7, (0, 165, 255), -1, cv2.LINE_AA)


def draw_status(frame, smooth_points, boundaries, tracking_source=""):
    # Show tracking source so user knows when tracker fallback is active
    text = (f"T{CURRENT_TRAJECTORY_ID:03d} | "
            f"pts={len(smooth_points)} | "
            f"seg_pts={len(boundaries)} | "
            f"src={tracking_source} | "
            "[s]save  [r]reset  [q]quit")
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE GRAPH WINDOW  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def render_live_graph(smooth_points, boundaries, start_point):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(LIVE_GRAPH_WIN_W / LIVE_GRAPH_DPI,
                          LIVE_GRAPH_WIN_H / LIVE_GRAPH_DPI),
                 dpi=LIVE_GRAPH_DPI)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    valid = [p for p in smooth_points if is_finite(p)]
    if len(valid) >= 2:
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        ax.plot(xs, ys, color="blue", linewidth=2.5, label="Trajectory", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
                   edgecolors="black", linewidths=0.7, label="End point", zorder=6)
        pad = GRAPH_PADDING_PIXELS
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)

    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.7, label="Start point", zorder=7)

    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red", edgecolors="black",
                   linewidths=0.7, label="Segment point", zorder=8)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="grey")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate(X_LABEL_TEXT,
                xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
                xytext=(6, -18), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="top")
    ax.annotate(Y_LABEL_TEXT,
                xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
                xytext=(-28, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom")
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="black", framealpha=1.0,
              fontsize=8, title="Legend", title_fontsize=9,
              borderpad=0.7, labelspacing=0.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.text(0.01, 0.01, FOOTER_TEXT, fontsize=6, color="black")

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = cv2.cvtColor(buf[..., :3], cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (LIVE_GRAPH_WIN_W, LIVE_GRAPH_WIN_H),
                     interpolation=cv2.INTER_AREA)
    plt.close(fig)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# FINAL HIGH-QUALITY GRAPH  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def save_final_graph(smooth_points, boundaries, start_point):
    valid = [p for p in smooth_points if is_finite(p)]
    if not valid:
        print("[Graph] No valid points — skipping graph save.", flush=True)
        return

    xs = [float(p[0]) for p in valid]
    ys = [float(p[1]) for p in valid]

    pad   = GRAPH_PADDING_PIXELS
    x_min = max(0, min(xs) - pad)
    x_max = max(xs) + pad
    y_min = max(0, min(ys) - pad)
    y_max = max(ys) + pad
    if x_max - x_min < 220:
        mid   = 0.5 * (x_min + x_max)
        x_min = max(0, mid - 110)
        x_max = mid + 110
    if y_max - y_min < 220:
        mid   = 0.5 * (y_min + y_max)
        y_min = max(0, mid - 110)
        y_max = mid + 110

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    ax.plot(xs, ys, color="blue", linewidth=2.8, label="Trajectory", zorder=3)

    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.8, label="Start point", zorder=6)

    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red",
                   edgecolors="black", linewidths=0.8,
                   label="Segment point", zorder=7)

    ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
               edgecolors="black", linewidths=0.8,
               label="End point", zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate(X_LABEL_TEXT,
                xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
                xytext=(6, -20), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="top", annotation_clip=False)
    ax.annotate(Y_LABEL_TEXT,
                xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
                xytext=(-30, 6), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="bottom", annotation_clip=False)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45, color="grey")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(TICK_LABEL_FONTSIZE)
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", length=6, width=1.4, direction="out")

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")

    legend = ax.legend(
        loc="upper right", frameon=True, fancybox=False,
        edgecolor="black", framealpha=1.0,
        fontsize=LEGEND_FONTSIZE, title="Legend",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        borderpad=0.8, labelspacing=0.7,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(1.1)

    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.92)
    fig.text(0.02, 0.02, FOOTER_TEXT, fontsize=8, color="black",
             wrap=True, ha="left", va="bottom")

    fig.savefig(final_image_path,   dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(preview_image_path, dpi=150,      bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Graph] Saved → {final_image_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON SAVE  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def save_framewise_csv(records, boundaries, start_frame):
    boundary_map = {int(b["frame_number"]): b for b in boundaries}
    with open(framewise_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_number", "segment_id", "segment_label",
            "tracking_source", "detected", "confidence",
            "x_raw", "y_raw", "x_smooth", "y_smooth",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "bbox_width", "bbox_height", "bbox_area", "aspect_ratio",
            "upper_anchor_ratio", "boundary_id", "boundary_cue", "is_boundary",
            "motion_start_frame",
        ])
        for r in records:
            fid = int(r["frame_number"])
            sid, slbl = seg_for_frame(fid, boundaries, start_frame)
            b = boundary_map.get(fid)
            w.writerow([
                fid, sid, slbl,
                r["tracking_source"], r["detected"], r["confidence"],
                r["x_raw"], r["y_raw"], r["x_smooth"], r["y_smooth"],
                r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"],
                r["bbox_width"], r["bbox_height"], r["bbox_area"], r["aspect_ratio"],
                UPPER_ANCHOR_RATIO,
                b["boundary_id"] if b else "",
                b["cue_type"]    if b else "none",
                bool(b),
                start_frame if start_frame is not None else "",
            ])


def save_boundary_csv(boundaries):
    with open(boundary_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary_id", "frame_number", "x", "y",
                    "direction_change_deg", "boundary_score", "cue_type"])
        for b in boundaries:
            w.writerow([b["boundary_id"], b["frame_number"], b["x"], b["y"],
                        b["direction_change"], b["boundary_score"], b["cue_type"]])


def save_segment_csv(segments):
    with open(segment_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "label", "start_frame", "end_frame", "duration_frames"])
        for s in segments:
            w.writerow([s["segment_id"], s["label"],
                        s["start_frame"], s["end_frame"], s["duration_frames"]])


def save_summary_csv(records, boundaries, segments, start_frame, start_point, total_frames):
    if not records:
        return
    valid = [r for r in records if not is_nan(r["x_smooth"]) and not is_nan(r["y_smooth"])]
    if not valid:
        return

    last        = valid[-1]
    end_frame   = int(last["frame_number"])
    traj_id_str = f"{RUN_NAME}_trajectory_{CURRENT_TRAJECTORY_ID:03d}"

    row = {
        "trajectory_id":      traj_id_str,
        "trajectory_number":  CURRENT_TRAJECTORY_ID,
        "target_class":       TARGET_CLASS,
        "camera_index":       ACTIVE_CAMERA_INDEX,
        "total_frames":       total_frames,
        "start_frame":        start_frame if start_frame is not None else "",
        "start_x":            float(start_point[0]) if is_finite(start_point) else "",
        "start_y":            float(start_point[1]) if is_finite(start_point) else "",
        "end_frame":          end_frame,
        "end_x":              float(last["x_smooth"]),
        "end_y":              float(last["y_smooth"]),
        "num_segment_points": len(boundaries),
        "num_segments":       len(segments),
        "boundary_frames":    ";".join(str(b["frame_number"]) for b in boundaries),
        "segment_ranges":     ";".join(
            f"{s['label']}:{s['start_frame']}-{s['end_frame']}" for s in segments),
        "output_folder":      OUTPUT_DIR,
        "final_graph":        final_image_path,
    }

    with open(trajectory_summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)

    master_exists = os.path.exists(master_summary_csv)
    with open(master_summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not master_exists: w.writeheader()
        w.writerow(row)

    print(f"[T{CURRENT_TRAJECTORY_ID:03d}] pts={len(valid)}  "
          f"boundaries={len(boundaries)}  segments={len(segments)}", flush=True)


def save_json_output(start_frame, start_point, boundaries, segments, total_frames):
    with open(json_path, "w") as f:
        json.dump({
            "trajectory_id": CURRENT_TRAJECTORY_ID,
            "camera_index":  ACTIVE_CAMERA_INDEX,
            "target_class":  TARGET_CLASS,
            "total_frames":  total_frames,
            "motion_start":  ({"frame_number": int(start_frame),
                               "x": float(start_point[0]),
                               "y": float(start_point[1])}
                              if start_frame is not None and is_finite(start_point)
                              else None),
            "boundaries":    boundaries,
            "segments":      segments,
            "output_files": {
                "framewise_csv":  framewise_csv_path,
                "boundary_csv":   boundary_csv_path,
                "segment_csv":    segment_csv_path,
                "summary_csv":    trajectory_summary_csv_path,
                "final_graph":    final_image_path,
                "preview_graph":  preview_image_path,
                "video":          output_video_path,
            },
        }, f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE COMPLETE TRAJECTORY  (unchanged from Code 1)
# ══════════════════════════════════════════════════════════════════════════════

def save_trajectory(state, width, height):
    if not state["records"]:
        print("[Save] Nothing to save.", flush=True)
        return

    total  = int(state["frame_number"])
    sf     = state["start_frame"]
    si     = state["start_index"]

    raw_b   = detect_boundaries(state["smooth_points"], state["frame_numbers"], si)
    final_b = clean_boundaries(raw_b, sf, total)
    final_s = make_segments(final_b, sf, total)

    sp = state["smooth_points"][si] if (si is not None and si < len(state["smooth_points"])) else None

    save_framewise_csv(state["records"], final_b, sf)
    save_boundary_csv(final_b)
    save_segment_csv(final_s)
    save_summary_csv(state["records"], final_b, final_s, sf, sp, total)
    save_json_output(sf, sp, final_b, final_s, total)
    save_final_graph(state["smooth_points"], final_b, sp)

    if state["video_writer"] is not None:
        state["video_writer"].release()
        state["video_writer"] = None

    print(f"[Save] Trajectory {CURRENT_TRAJECTORY_ID} → {OUTPUT_DIR}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def fresh_state():
    return {
        "frame_number":       0,
        "records":            [],
        "raw_points":         [],
        "smooth_points":      [],
        "frame_numbers":      [],
        "last_anchor":        None,
        "last_bbox":          None,
        "last_smooth_bbox":   None,   # ← NEW: EMA-smoothed bbox
        "last_metrics":       None,
        "prev_ema":           None,
        "hold_counter":       0,
        "start_frame":        None,
        "start_index":        None,
        "video_writer":       None,
        # ── NEW tracker state ──────────────────────────────────────────────
        "tracker":            None,
        "tracker_only_count": 0,
        "last_tracking_src":  "lost",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CURRENT_TRAJECTORY_ID

    model = None
    for candidate in MODEL_PATH_CANDIDATES:
        if os.path.exists(candidate) or candidate == "yolov8n.pt":
            print(f"[YOLO] Loading weights from: {candidate}", flush=True)
            try:
                model = YOLO(candidate)
                break
            except Exception as e:
                print(f"[YOLO] Failed to load {candidate}: {e}", flush=True)

    if model is None:
        print("[CRITICAL ERROR] Could not initialise YOLOv8.", flush=True)
        return

    cap = cv2.VideoCapture(ACTIVE_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[CRITICAL ERROR] Could not open camera {ACTIVE_CAMERA_INDEX}.", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    ret, initial_frame = cap.read()
    if not ret:
        print("[CRITICAL ERROR] Camera returned empty frame.", flush=True)
        cap.release()
        return
    h, w = initial_frame.shape[:2]

    state = fresh_state()
    print("\n>>> PIPELINE RUNNING. Focus the window and use keyboard to interact.", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline] Frame grab failure or stream ended.", flush=True)
                break

            state["frame_number"] += 1
            current_fid = state["frame_number"]

            # ── Video writer init ──────────────────────────────────────────
            if SAVE_ANNOTATED_VIDEO and state["video_writer"] is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                state["video_writer"] = cv2.VideoWriter(
                    output_video_path, fourcc, CAMERA_FPS, (w, h))

            # ══════════════════════════════════════════════════════════════
            # DETECTION & TRACKING  (rewritten with Code 2 strategies)
            # ══════════════════════════════════════════════════════════════

            detections    = []
            raw_bbox      = None
            raw_pt        = None
            source_str    = "lost"
            conf_val      = 0.0

            # ── 1. Run YOLO on schedule ────────────────────────────────────
            run_yolo = (current_fid % YOLO_EVERY_N_FRAMES == 0) or (state["last_anchor"] is None)
            if run_yolo:
                results = model.predict(frame, conf=CONF_THRESHOLD,
                                        imgsz=YOLO_IMGSZ, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if model.names[cls_id] == TARGET_CLASS:
                        bbox_raw  = box.xyxy[0].cpu().numpy()
                        bbox_clip = clip_bbox(bbox_raw, w, h)
                        detections.append({
                            "bbox":       bbox_clip,
                            "anchor":     upper_anchor(bbox_clip),
                            "confidence": float(box.conf[0]),
                        })

            # ── 2. Data association: nearest-to-last-anchor (FIX 1) ────────
            chosen = best_detection(detections, state["last_anchor"])

            if chosen is not None:
                # YOLO found it — reinitialise the OpenCV tracker
                raw_bbox   = chosen["bbox"]
                source_str = "yolo"
                conf_val   = chosen["confidence"]
                state["hold_counter"]       = 0
                state["tracker_only_count"] = 0

                if USE_OPENCV_TRACKER_WHEN_YOLO_MISSES:
                    state["tracker"] = create_cv_tracker()
                    if state["tracker"] is not None:
                        try:
                            state["tracker"].init(frame, xyxy_to_xywh(raw_bbox))
                        except Exception:
                            state["tracker"] = None

            # ── 3. OpenCV tracker fallback when YOLO misses (FIX 3) ────────
            elif (USE_OPENCV_TRACKER_WHEN_YOLO_MISSES
                  and state["tracker"] is not None
                  and state["tracker_only_count"] < MAX_TRACKER_ONLY_FRAMES):
                try:
                    ok, tracked_xywh = state["tracker"].update(frame)
                except Exception:
                    ok = False

                if ok:
                    x1, y1, x2, y2 = xywh_to_xyxy(tracked_xywh)
                    x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), w, h)
                    raw_bbox = (x1, y1, x2, y2)
                    source_str = "tracker"
                    conf_val   = 0.0
                    state["tracker_only_count"] += 1
                else:
                    state["tracker"] = None
                    source_str = "lost"

            # ── 4. Legacy hold-position fallback (original Code 1 safety net)
            elif (state["last_anchor"] is not None
                  and state["hold_counter"] < MAX_HOLD_FRAMES
                  and not USE_OPENCV_TRACKER_WHEN_YOLO_MISSES):
                state["hold_counter"] += 1
                raw_pt     = state["last_anchor"]
                source_str = "hold_position"
                conf_val   = 0.0

            # ── 5. Compute anchor from (optionally smoothed) bbox ──────────
            if raw_bbox is not None:
                # Apply EMA smoothing to bbox before computing anchor (FIX 2)
                smoothed_b = smooth_bbox(
                    raw_bbox,
                    state["last_smooth_bbox"],
                    BBOX_EMA_ALPHA,
                )
                state["last_smooth_bbox"] = smoothed_b
                state["last_bbox"]        = raw_bbox          # keep raw for CSV
                state["last_metrics"]     = bbox_metrics(raw_bbox)

                # Anchor from SMOOTHED bbox → stable through tilt/reshape
                raw_pt = upper_anchor(smoothed_b)
                state["last_anchor"] = raw_pt

            state["last_tracking_src"] = source_str

            # ══════════════════════════════════════════════════════════════
            # POINT ACCUMULATION & SMOOTHING  (unchanged from Code 1)
            # ══════════════════════════════════════════════════════════════

            if raw_pt is not None:
                ema_pt = ema_filter(raw_pt, state["prev_ema"], EMA_ALPHA)
                state["prev_ema"] = ema_pt
                state["raw_points"].append(raw_pt)

                smooth_pt = savgol_smooth(state["raw_points"])
                state["smooth_points"].append(smooth_pt)
                state["frame_numbers"].append(current_fid)

                if state["start_frame"] is None:
                    s_idx, s_fnum = detect_start(state["smooth_points"],
                                                  state["frame_numbers"])
                    if s_idx is not None:
                        state["start_index"] = s_idx
                        state["start_frame"] = s_fnum

                metrics     = state["last_metrics"] if state["last_metrics"] else {
                    "bbox_width": "", "bbox_height": "", "bbox_area": "", "aspect_ratio": ""}
                bbox_coords = state["last_bbox"] if state["last_bbox"] is not None else ["", "", "", ""]

                state["records"].append({
                    "frame_number":   current_fid,
                    "tracking_source": source_str,
                    "detected":       (chosen is not None),
                    "confidence":     conf_val,
                    "x_raw":   raw_pt[0], "y_raw":    raw_pt[1],
                    "x_smooth": smooth_pt[0], "y_smooth": smooth_pt[1],
                    "bbox_x1": bbox_coords[0], "bbox_y1": bbox_coords[1],
                    "bbox_x2": bbox_coords[2], "bbox_y2": bbox_coords[3],
                    **metrics,
                })
            else:
                state["records"].append({
                    "frame_number":   current_fid,
                    "tracking_source": source_str,
                    "detected":       False,
                    "confidence":     0.0,
                    "x_raw": float("nan"), "y_raw":    float("nan"),
                    "x_smooth": float("nan"), "y_smooth": float("nan"),
                    "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                    "bbox_width": "", "bbox_height": "", "bbox_area": "", "aspect_ratio": "",
                })

            # ── Segmentation for live render ───────────────────────────────
            current_boundaries = detect_boundaries(
                state["smooth_points"], state["frame_numbers"], state["start_index"])

            # ── Video overlay ──────────────────────────────────────────────
            annotated_frame = frame.copy()
            start_pt_val = (state["smooth_points"][state["start_index"]]
                            if state["start_index"] is not None else None)
            draw_overlay(annotated_frame, state["smooth_points"],
                         current_boundaries, start_pt_val, state["start_frame"])
            draw_status(annotated_frame, state["smooth_points"],
                        current_boundaries, state["last_tracking_src"])

            if state["video_writer"] is not None:
                state["video_writer"].write(annotated_frame)

            if SHOW_WINDOWS:
                cv2.imshow("Real-Time YOLOv8 Trajectory Tracker", annotated_frame)

                if (SHOW_LIVE_GRAPH_WINDOW
                        and current_fid % LIVE_GRAPH_UPDATE_EVERY_N_FRAMES == 0):
                    graph_img = render_live_graph(
                        state["smooth_points"], current_boundaries, start_pt_val)
                    cv2.imshow("Upper-Anchor Trajectory Graph", graph_img)

            # ── Keyboard controls ──────────────────────────────────────────
            key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF
            if key == ord("q"):
                print("\n[Control] Quit — saving...", flush=True)
                save_trajectory(state, w, h)
                break
            elif key == ord("s"):
                print(f"\n[Control] Save trajectory {CURRENT_TRAJECTORY_ID} → next.", flush=True)
                save_trajectory(state, w, h)
                CURRENT_TRAJECTORY_ID += 1
                set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)
                state = fresh_state()
            elif key == ord("r"):
                print(f"\n[Control] Reset trajectory {CURRENT_TRAJECTORY_ID}.", flush=True)
                if state["video_writer"] is not None:
                    state["video_writer"].release()
                state = fresh_state()

    except KeyboardInterrupt:
        print("\n[Control] KeyboardInterrupt.", flush=True)
        save_trajectory(state, w, h)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSCRIPT EXECUTION TERMINATED CLEANLY.", flush=True)


if __name__ == "__main__":
    main()