"""
Real-Time YOLOv8 Camera Object-Trajectory Segmentation

Graph style  : matches the reference image exactly
    Title    : Object Upper-Anchor Trajectory in Image Coordinate Space
    X-label  : X-coordinate of Object Upper-Anchor Point (pixels)
    Y-label  : Y-coordinate of Object Upper-Anchor Point (pixels)
    Legend   : Trajectory (blue line) | Start point (green) |
               Segment point (red)    | End point (orange)
    Footer   : coordinate-system note below the axes

CSV outputs  :
    framewise_data.csv
    boundary_points.csv
    temporal_segments.csv
    trajectory_summary.csv
    all_trajectories_summary.csv   ← master, one row per trajectory

Controls:
    s  =  save current trajectory → start next trajectory
    r  =  reset current trajectory without saving
    q  =  save current trajectory → exit

Install:
    pip install ultralytics opencv-python numpy scipy matplotlib
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


# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  ── edit only this section
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX       = 1          # try 0 if 1 does not work
TARGET_CLASS       = "bottle"   # any COCO class name
CONF_THRESHOLD     = 0.35
YOLO_IMGSZ         = 320

UPPER_ANCHOR_RATIO = 0.22       # fraction of bbox height from top

SHOW_WINDOWS           = True
SHOW_CURRENT_POINT     = True   # orange dot on live camera feed
SAVE_ANNOTATED_VIDEO   = True
SHOW_LIVE_GRAPH_WINDOW = True   # separate matplotlib-rendered window

PLAYBACK_DELAY_MS = 1

# Output root – auto-selects D:\ on Windows if present, else Desktop
if os.path.exists("D:\\"):
    BASE_OUTPUT_DIR = r"D:\trajectory_output"
else:
    BASE_OUTPUT_DIR = os.path.join(
        os.path.expanduser("~"), "Desktop", "trajectory_output"
    )

# ── YOLO model search order ──────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = os.getcwd()

MODEL_PATH_CANDIDATES = [
    os.path.join(CURRENT_DIR, "yolo-Weights", "yolov8n.pt"),
    os.path.join(SCRIPT_DIR,  "yolo-Weights", "yolov8n.pt"),
    r"C:\Users\srish\OneDrive\Documents\yolo\yolo-Weights\yolov8n.pt",
    "yolov8n.pt",   # falls back to auto-download
]

# ── Camera capture settings ──────────────────────────────────────────────────
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

YOLO_EVERY_N_FRAMES = 1
MAX_HOLD_FRAMES     = 5

# ── Smoothing ────────────────────────────────────────────────────────────────
SAVGOL_WINDOW    = 11
SAVGOL_POLYORDER = 2
EMA_ALPHA        = 0.35

# ── Start detection ──────────────────────────────────────────────────────────
START_SPEED_PIXELS_PER_FRAME      = 0.35
START_MOTION_CONSECUTIVE_FRAMES   = 3
START_MOTION_DISTANCE_PIXELS      = 4.0
START_REST_BACKTRACK_FRAMES       = 1
FORCE_START_AFTER_VALID_POINTS    = 5

# ── Segmentation ─────────────────────────────────────────────────────────────
MIN_BEND_ANGLE_DEG              = 45.0
BEND_WINDOW_FRAMES              = 5
MIN_BOUNDARY_GAP_FRAMES         = 10
FIRST_BOUNDARY_MIN_GAP_FRAMES   = 3
MIN_SEGMENT_DISPLACEMENT_PIXELS = 6.0
MIN_SEGMENT_LENGTH_FRAMES       = 5
MAX_BOUNDARIES                  = None   # None = unlimited

# ── Final graph (saved PNG) ───────────────────────────────────────────────────
FIGURE_SIZE          = (10, 7)
FIGURE_DPI           = 220
SAVE_DPI             = 350
GRAPH_PADDING_PIXELS = 100
MARKER_SIZE          = 90          # scatter dot size (points²)

TITLE_TEXT   = "Object Upper-Anchor Trajectory in Image Coordinate Space"
X_LABEL_TEXT = "X"   # placed at the right end of the x-axis
Y_LABEL_TEXT = "Y"   # placed at the top end of the y-axis
FOOTER_TEXT   = ("Coordinate system: image pixel coordinates obtained from the "
                 "detected object upper-anchor point. "
                 "Image-coordinate Y increases downward.")

AXIS_LABEL_FONTSIZE   = 13
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE   = 11
LEGEND_FONTSIZE       = 11
LEGEND_TITLE_FONTSIZE = 12

# ── Live graph window (OpenCV BGR render of matplotlib figure) ────────────────
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

# (populated by set_trajectory_output_paths)
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
    x1, y1, x2, y2 = bbox
    bh = max(1.0, float(y2 - y1))
    return float((x1 + x2) / 2.0), float(y1 + UPPER_ANCHOR_RATIO * bh)


def best_detection(detections, last_anchor):
    if not detections:
        return None
    if last_anchor is None:
        return max(detections, key=lambda d: d["confidence"])
    return min(detections, key=lambda d: euclidean(d["anchor"], last_anchor))


def angle_between(v1, v2):
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_v = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_v)))


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING
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
# START DETECTION
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
        return None, None
    ri   = max(0, fi - START_REST_BACKTRACK_FRAMES)
    li   = min(len(pts) - 1, ri + START_MOTION_CONSECUTIVE_FRAMES + 3)
    disp = float(np.linalg.norm(pts[li] - pts[ri]))
    if disp < START_MOTION_DISTANCE_PIXELS:
        return None, None
    return ri, int(frame_numbers[ri])


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
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
    cleaned = []
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
# CAMERA FEED OVERLAY  (drawn on the CV window)
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame, smooth_points, boundaries, start_point, start_frame):
    # Blue trajectory line
    valid = [(int(p[0]), int(p[1])) for p in smooth_points if is_finite(p)]
    for i in range(1, len(valid)):
        cv2.line(frame, valid[i - 1], valid[i], (255, 0, 0), 3, cv2.LINE_AA)

    # Green start dot
    if start_point is not None and start_frame is not None:
        cv2.circle(frame, (int(start_point[0]), int(start_point[1])),
                   9, (0, 255, 0), -1, cv2.LINE_AA)

    # Red segment-boundary dots
    for b in boundaries:
        cv2.circle(frame, (int(b["x"]), int(b["y"])),
                   10, (0, 0, 255), -1, cv2.LINE_AA)

    # Orange current dot
    if SHOW_CURRENT_POINT and smooth_points:
        p = smooth_points[-1]
        if is_finite(p):
            cv2.circle(frame, (int(p[0]), int(p[1])),
                       7, (0, 165, 255), -1, cv2.LINE_AA)


def draw_status(frame, smooth_points, boundaries):
    text = (f"T{CURRENT_TRAJECTORY_ID:03d} | "
            f"pts={len(smooth_points)} | "
            f"boundaries={len(boundaries)} | "
            "[s]save  [r]reset  [q]quit")
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE GRAPH WINDOW  (matplotlib → OpenCV image)
# ══════════════════════════════════════════════════════════════════════════════

def render_live_graph(smooth_points, boundaries, start_point):
    """Return a BGR numpy image matching the reference graph style."""
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

        # End point (orange)
        ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
                   edgecolors="black", linewidths=0.7, label="End point", zorder=6)

        pad = GRAPH_PADDING_PIXELS
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)   # Y inverted

    # Start point (green)
    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.7, label="Start point", zorder=7)

    # Segment / boundary points (red)
    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red", edgecolors="black",
                   linewidths=0.7, label="Segment point", zorder=8)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="grey")
    ax.invert_yaxis()

    # ── X label at right end of x-axis, Y label at top end of y-axis ─────────
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate(
        X_LABEL_TEXT,
        xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
        xytext=(6, -18), textcoords="offset points",
        fontsize=9, fontweight="bold", ha="left", va="top",
    )
    ax.annotate(
        Y_LABEL_TEXT,
        xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
        xytext=(-28, 6), textcoords="offset points",
        fontsize=9, fontweight="bold", ha="left", va="bottom",
    )

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
# FINAL HIGH-QUALITY GRAPH  (saved to disk, matches reference image exactly)
# ══════════════════════════════════════════════════════════════════════════════

def save_final_graph(smooth_points, boundaries, start_point):
    valid = [p for p in smooth_points if is_finite(p)]
    if not valid:
        print("[Graph] No valid points — skipping graph save.", flush=True)
        return

    xs = [float(p[0]) for p in valid]
    ys = [float(p[1]) for p in valid]

    # Dynamic axis limits with padding
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

    # ── Trajectory line ──────────────────────────────────────────────────────
    ax.plot(xs, ys, color="blue", linewidth=2.8,
            label="Trajectory", zorder=3)

    # ── Start point (green) ──────────────────────────────────────────────────
    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.8, label="Start point", zorder=6)

    # ── Segment / boundary points (red) ─────────────────────────────────────
    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red",
                   edgecolors="black", linewidths=0.8,
                   label="Segment point", zorder=7)

    # ── End point (orange) ───────────────────────────────────────────────────
    ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
               edgecolors="black", linewidths=0.8,
               label="End point", zorder=6)

    # ── Axes limits and inversion ────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()          # image Y increases downward → 0 at top
    ax.set_aspect("equal", adjustable="box")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title(TITLE_TEXT, fontsize=14, fontweight="bold", pad=12)

    # ── X label at right end of x-axis, Y label at top end of y-axis ────────
    ax.set_xlabel("")
    ax.set_ylabel("")
    # "X" just to the right of the x-axis right end
    ax.annotate(
        X_LABEL_TEXT,
        xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
        xytext=(6, -20), textcoords="offset points",
        fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
        ha="left", va="top", annotation_clip=False,
    )
    # "Y" just above the top of the y-axis
    ax.annotate(
        Y_LABEL_TEXT,
        xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
        xytext=(-30, 6), textcoords="offset points",
        fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
        ha="left", va="bottom", annotation_clip=False,
    )

    # ── Grid (dashed, light grey, as in reference) ───────────────────────────
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45, color="grey")

    # ── Tick labels ──────────────────────────────────────────────────────────
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(TICK_LABEL_FONTSIZE)
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", length=6, width=1.4, direction="out")

    # ── Spines ───────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")

    # ── Legend (upper right, white background, "Legend" title) ──────────────
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=LEGEND_FONTSIZE,
        title="Legend",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        borderpad=0.8,
        labelspacing=0.7,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(1.1)

    # ── Footer note (bottom-left, below axes) ────────────────────────────────
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.92)
    fig.text(0.02, 0.02, FOOTER_TEXT, fontsize=8, color="black",
             wrap=True, ha="left", va="bottom")

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(final_image_path,   dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(preview_image_path, dpi=150,      bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Graph] Saved → {final_image_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON SAVE
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

    last         = valid[-1]
    end_frame    = int(last["frame_number"])
    traj_id_str  = f"{RUN_NAME}_trajectory_{CURRENT_TRAJECTORY_ID:03d}"

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
# SAVE COMPLETE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_trajectory(state, width, height):
    if not state["records"]:
        print("[Save] Nothing to save.", flush=True)
        return

    total  = int(state["frame_number"])
    sf     = state["start_frame"]
    si     = state["start_index"]

    # Re-compute final clean boundaries and segments
    raw_b  = detect_boundaries(state["smooth_points"], state["frame_numbers"], si)
    final_b = clean_boundaries(raw_b, sf, total)
    final_s = make_segments(final_b, sf, total)

    sp = state["smooth_points"][si] if (si is not None and si < len(state["smooth_points"])) else None

    save_framewise_csv(state["records"], final_b, sf)
    save_boundary_csv(final_b)
    save_segment_csv(final_s)
    save_summary_csv(state["records"], final_b, final_s, sf, sp, total)
    save_json_output(sf, sp, final_b, final_s, total)
    save_final_graph(state["smooth_points"], final_b, sp)

    print(f"[Save] Trajectory {CURRENT_TRAJECTORY_ID} → {OUTPUT_DIR}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

def fresh_state():
    return {
        "frame_number":    0,
        "records":         [],
        "raw_points":      [],
        "smooth_points":   [],
        "frame_numbers":   [],
        "last_anchor":     None,
        "last_bbox":       None,
        "last_metrics":    None,
        "prev_ema":        None,
        "missed":          0,
        "start_index":     None,
        "start_frame":     None,
        "start_point":     None,
        "boundaries":      [],
        "segments":        [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA + MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print("Loading YOLOv8 model…", flush=True)
    from ultralytics import YOLO

    model_path = next(
        (c for c in MODEL_PATH_CANDIDATES
         if c == "yolov8n.pt" or os.path.exists(c)),
        "yolov8n.pt"
    )
    print(f"  Using: {model_path}", flush=True)
    model = YOLO(model_path)
    try: model.fuse()
    except Exception: pass

    target_id = next(
        (cid for cid, name in model.names.items()
         if name.lower() == TARGET_CLASS.lower()),
        None
    )
    if target_id is None:
        raise ValueError(
            f"Class '{TARGET_CLASS}' not in model. "
            f"Available: {list(model.names.values())}"
        )
    print(f"  Target class '{TARGET_CLASS}' → id {target_id}", flush=True)
    return model, target_id


def open_camera():
    global ACTIVE_CAMERA_INDEX
    system = platform.system().lower()
    if "darwin"  in system: backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif "windows" in system: backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:                   backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for idx in dict.fromkeys([CAMERA_INDEX, 0, 1, 2, 3]):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                ret, frame = cap.read()
                if ret and frame is not None:
                    ACTIVE_CAMERA_INDEX = idx
                    print(f"Camera opened: index={idx}", flush=True)
                    return cap
                cap.release()
    raise RuntimeError("Cannot open any camera. Try changing CAMERA_INDEX.")


def open_writer(w, h, fps):
    if not SAVE_ANNOTATED_VIDEO:
        return None
    return cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    model, target_id = load_model()
    cap              = open_camera()

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS

    out   = open_writer(W, H, fps)
    state = fresh_state()

    live_graph_img = None

    if SHOW_WINDOWS:
        cv2.namedWindow("Trajectory Cam",  cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trajectory Cam", 900, 650)
        if SHOW_LIVE_GRAPH_WINDOW:
            cv2.namedWindow("Live Graph",  cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Live Graph", LIVE_GRAPH_WIN_W, LIVE_GRAPH_WIN_H)

    print("\n[Ready]  s=save+next  r=reset  q=save+quit", flush=True)
    print(f"[Output] {RUN_ROOT_DIR}", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame read failed — retrying…", flush=True)
                continue

            state["frame_number"] += 1
            fn = state["frame_number"]

            # ── YOLO ──────────────────────────────────────────────────────────
            detections = []
            if fn % YOLO_EVERY_N_FRAMES == 0:
                results = model(frame, conf=CONF_THRESHOLD,
                                imgsz=YOLO_IMGSZ, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) != target_id:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bb  = clip_bbox((int(x1), int(y1), int(x2), int(y2)), W, H)
                        anc = upper_anchor(bb)
                        detections.append({
                            "bbox": bb, "anchor": anc,
                            "confidence": conf, "metrics": bbox_metrics(bb),
                        })

            sel = best_detection(detections, state["last_anchor"])

            # ── Tracking ──────────────────────────────────────────────────────
            has_pt   = False
            detected = False
            src      = "none"
            conf_v   = np.nan
            raw_x = raw_y = np.nan
            bbox_v = metrics_v = None

            if sel is not None:
                detected = True
                src      = "yolo"
                bbox_v   = sel["bbox"]
                conf_v   = sel["confidence"]
                raw_x, raw_y = sel["anchor"]
                metrics_v    = sel["metrics"]
                state["last_anchor"]  = (float(raw_x), float(raw_y))
                state["last_bbox"]    = bbox_v
                state["last_metrics"] = metrics_v
                state["missed"]       = 0
                has_pt = True
            elif state["last_anchor"] is not None and state["missed"] < MAX_HOLD_FRAMES:
                src      = "short_hold"
                raw_x, raw_y = state["last_anchor"]
                bbox_v   = state["last_bbox"]
                metrics_v = state["last_metrics"]
                state["missed"] += 1
                has_pt = True
            else:
                src = "lost"
                state["missed"] += 1

            # ── Smooth & record ───────────────────────────────────────────────
            sx = sy = np.nan
            if has_pt:
                state["raw_points"].append((float(raw_x), float(raw_y)))
                sg   = savgol_smooth(state["raw_points"])
                ema  = ema_filter(sg, state["prev_ema"], EMA_ALPHA)
                state["prev_ema"] = ema
                sx, sy = float(ema[0]), float(ema[1])
                state["smooth_points"].append((sx, sy))
                state["frame_numbers"].append(fn)

                # Start detection
                if state["start_frame"] is None:
                    si, sf = detect_start(state["smooth_points"], state["frame_numbers"])
                    if si is not None:
                        state["start_index"] = si
                        state["start_frame"] = sf
                        state["start_point"] = state["smooth_points"][si]
                    elif len(state["smooth_points"]) >= FORCE_START_AFTER_VALID_POINTS:
                        state["start_index"] = 0
                        state["start_frame"] = int(state["frame_numbers"][0])
                        state["start_point"] = state["smooth_points"][0]

                # Live boundary update
                if state["start_frame"] is not None:
                    raw_b  = detect_boundaries(
                        state["smooth_points"], state["frame_numbers"], state["start_index"])
                    state["boundaries"] = clean_boundaries(raw_b, state["start_frame"], fn)
                    state["segments"]   = make_segments(
                        state["boundaries"], state["start_frame"], fn)

            if metrics_v:
                bw = metrics_v["bbox_width"];  bh = metrics_v["bbox_height"]
                ba = metrics_v["bbox_area"];   ar = metrics_v["aspect_ratio"]
            else:
                bw = bh = ba = ar = np.nan
            if bbox_v:
                bx1, by1, bx2, by2 = bbox_v
            else:
                bx1 = by1 = bx2 = by2 = ""

            state["records"].append({
                "frame_number": fn,
                "tracking_source": src,
                "detected": detected,
                "confidence": conf_v,
                "x_raw": raw_x, "y_raw": raw_y,
                "x_smooth": sx, "y_smooth": sy,
                "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
                "bbox_width": bw, "bbox_height": bh, "bbox_area": ba, "aspect_ratio": ar,
            })

            # ── Display ───────────────────────────────────────────────────────
            disp = frame.copy()
            draw_overlay(disp, state["smooth_points"],
                         state["boundaries"], state["start_point"], state["start_frame"])
            draw_status(disp, state["smooth_points"], state["boundaries"])

            if out is not None:
                out.write(disp)

            if SHOW_WINDOWS:
                cv2.imshow("Trajectory Cam", disp)

                # Live graph (re-render every N frames)
                if (SHOW_LIVE_GRAPH_WINDOW
                        and len(state["smooth_points"]) >= 2
                        and fn % LIVE_GRAPH_UPDATE_EVERY_N_FRAMES == 0):
                    live_graph_img = render_live_graph(
                        state["smooth_points"],
                        state["boundaries"],
                        state["start_point"],
                    )
                if live_graph_img is not None and SHOW_LIVE_GRAPH_WINDOW:
                    cv2.imshow("Live Graph", live_graph_img)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF

            if key == SAVE_AND_NEXT_KEY:          # s
                if out is not None: out.release(); out = None
                save_trajectory(state, W, H)
                CURRENT_TRAJECTORY_ID_next = CURRENT_TRAJECTORY_ID + 1
                set_trajectory_output_paths(CURRENT_TRAJECTORY_ID_next)
                state          = fresh_state()
                live_graph_img = None
                out            = open_writer(W, H, fps)
                print(f"[Ready] Trajectory {CURRENT_TRAJECTORY_ID} started.", flush=True)

            elif key == ord("r"):                 # r
                if out is not None: out.release()
                state          = fresh_state()
                live_graph_img = None
                out            = open_writer(W, H, fps)
                print("[Reset] Trajectory reset (not saved).", flush=True)

            elif key == ord("q"):                 # q
                break

    except KeyboardInterrupt:
        print("[Interrupt] Stopping…", flush=True)

    finally:
        cap.release()
        if out is not None: out.release()
        if SHOW_WINDOWS: cv2.destroyAllWindows()
        save_trajectory(state, W, H)
        print(f"\n[Done] Output root: {RUN_ROOT_DIR}", flush=True)


if __name__ == "__main__":
    main()