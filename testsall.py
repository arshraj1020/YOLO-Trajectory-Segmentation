"""
Real-Time YOLOv8-Seg Bottle-Cap Trajectory Tracker  ── v3 (production)
═══════════════════════════════════════════════════════════════════════════════

ROOT-CAUSE FIXES vs Code 1 & Code 2
────────────────────────────────────
CODE 1 PROBLEMS FIXED:
  ✗ AABB upper-30% anchor breaks at any tilt  →  OBB cap-end anchor
  ✗ No rotation awareness at all              →  minAreaRect + mask contour
  ✗ EMA on bbox coords (wrong level)          →  EMA on anchor POINT instead

CODE 2 PROBLEMS FIXED:
  ✗ Stateless cap-end selection (flips at 90°/180°)
      →  TEMPORAL CAP MEMORY: once cap end established, subsequent frames
         enforce continuity via long-axis dot-product. Cap can only change
         if dot-product < 0 AND the flip is confirmed for N consecutive frames.
  ✗ y-tie-break fails when bottle is horizontal or inverted
      →  Primary: cross-section width at multiple depths (5%, 15%, 25%)
         Secondary: temporal dot-product continuity  (the real fix)
         Tertiary: y-position only as last resort
  ✗ cross-section sampled only at t=0.05 → noisy
      →  Multi-depth average: t ∈ {0.05, 0.12, 0.22}
  ✗ CONF_THRESHOLD=0.35 → spurious detections
      →  0.55 default; tune per environment
  ✗ Direction-change detector has no hysteresis flush on tracker frames
      →  flush() called on save/quit/reset

TRACKING ARCHITECTURE (new):
  1. YOLOv8n-seg → instance mask → minAreaRect OBB → long axis → two ends
  2. Cap end selected by: multi-depth cross-section narrowness
  3. TEMPORAL GUARD: cap_axis_memory stores last confirmed long-axis vector.
     Next frame: if new_long_axis · old_long_axis < COS_FLIP_THRESHOLD,
     a potential flip is staged. Only committed after FLIP_CONFIRM_FRAMES
     consecutive frames all agree the flip is real.
     This eliminates single-frame jitter flips entirely.
  4. Tracker fallback (CSRT→KCF→MIL): inherits last known cap direction,
     uses AABB heuristic but biased by last axis memory.
  5. Anchor EMA applied to the final (x,y) anchor point (not bbox coords).

SEGMENTATION:
  Multi-scale hysteresis direction-change detector from Code 2, retained
  exactly. No changes needed there.

GRAPH / CSV / JSON:
  Identical schema to Code 2.  Drop-in replacement.

USAGE:
    python bottle_tracker_v3.py
    [s] = save trajectory, start next
    [r] = reset without saving
    [q] = save and quit

REQUIREMENTS:
    pip install ultralytics opencv-python-headless numpy scipy matplotlib
    Place yolov8n-seg.pt in ./yolo-Weights/   (auto-downloaded if absent)
"""

print("BOTTLE TRACKER v3 STARTING", flush=True)

import os, csv, json, math
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not found – Savgol disabled, EMA-only smoothing.", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  USER SETTINGS  ── edit this section only
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX   = 1
TARGET_CLASS   = "bottle"
CONF_THRESHOLD = 0.45       # raise if spurious detections; lower if missed
YOLO_IMGSZ     = 640      # inference resolution

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# ── Rotation-robust cap tracking ─────────────────────────────────────────────
# Cosine threshold for long-axis continuity guard.
# If dot(new_axis, prev_axis) < this value the axis has flipped >90°.
# Value range: -1.0 … 1.0.   Recommended: -0.15
COS_FLIP_THRESHOLD   = 0.20

# How many consecutive frames must agree on a flip before it is accepted.
# Increase to 5-6 for very slow pours; decrease to 2 for fast flicks.
FLIP_CONFIRM_FRAMES  = 3

# Multi-depth cross-section sampling fractions (distance from end as fraction
# of OBB length).  More values = more robust; diminishing returns past 4.
CROSS_SECTION_DEPTHS = [0.04, 0.10, 0.20, 0.30]

# Min pixel area of mask contour to be considered a valid bottle
MIN_MASK_AREA_PX = 80

# ── OpenCV tracker fallback ───────────────────────────────────────────────────
USE_CV_TRACKER        = True
MAX_TRACKER_FRAMES    = 10  # give up after this many consecutive tracker-only frames

# ── Smoothing ─────────────────────────────────────────────────────────────────
SAVGOL_WINDOW    = 9
SAVGOL_POLYORDER = 2
EMA_ALPHA        = 0.50     # applied to anchor point (not bbox)

# ── Start detection ───────────────────────────────────────────────────────────
PRE_START_BUFFER_FRAMES       = 35
START_LOOKBACK_FRAMES         = 6
START_DISPLACEMENT_PIXELS     = 8.0
START_MIN_MEDIAN_SPEED_PIXELS = 1.0

# ── Multi-scale direction-change segmentation ─────────────────────────────────
DIRECTION_WINDOWS         = [4, 8, 12]
ANGLE_CHANGE_THRESHOLD    = 28.0
ANGLE_RESET_THRESHOLD     = 12.0
MIN_VECTOR_LENGTH_PIXELS  = 5.0
MIN_BOUNDARY_GAP_FRAMES   = 14
MIN_SEGMENT_LENGTH_FRAMES = 12
MAX_BOUNDARIES            = None

# ── Display & output ──────────────────────────────────────────────────────────
SHOW_WINDOWS              = True
SHOW_CURRENT_POINT        = True
SAVE_ANNOTATED_VIDEO      = True
SHOW_LIVE_GRAPH_WINDOW    = True
PLAYBACK_DELAY_MS         = 1
LIVE_GRAPH_UPDATE_EVERY_N = 5
LIVE_GRAPH_WIN_W          = 900
LIVE_GRAPH_WIN_H          = 620
LIVE_GRAPH_DPI            = 130

FIGURE_SIZE           = (10, 7)
FIGURE_DPI            = 220
SAVE_DPI              = 350
GRAPH_PADDING_PIXELS  = 100
MARKER_SIZE           = 90
X_LABEL_TEXT          = "X"
Y_LABEL_TEXT          = "Y"
FOOTER_TEXT           = ""
AXIS_LABEL_FONTSIZE   = 13
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE   = 11
LEGEND_FONTSIZE       = 11
LEGEND_TITLE_FONTSIZE = 12

# ── OBB overlay colours ───────────────────────────────────────────────────────
OBB_COLOR     = (0, 200, 255)    # cyan  – OBB outline
CAP_DOT_COLOR = (255, 0, 255)    # magenta – cap end dot
TRJ_COLOR     = (255, 80, 0)     # blue-ish – trajectory line

# ── Output root ───────────────────────────────────────────────────────────────
if os.path.exists("D:\\"):
    BASE_OUTPUT_DIR = r"D:\trajectory_output"
else:
    BASE_OUTPUT_DIR = os.path.join(os.path.expanduser("~"),
                                   "Desktop", "trajectory_output")

# ── YOLO model search order ───────────────────────────────────────────────────
_CWD  = os.getcwd()
_SDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else _CWD
MODEL_PATH_CANDIDATES = [
    os.path.join(_CWD,  "yolo-Weights", "yolov8s-seg.pt"),
    os.path.join(_SDIR, "yolo-Weights", "yolov8s-seg.pt"),
    os.path.join(_CWD,  "yolo-Weights", "yolov8n-seg.pt"),
    os.path.join(_SDIR, "yolo-Weights", "yolov8n-seg.pt"),
    "yolov8s-seg.pt",
    "yolov8n-seg.pt",
]


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT PATH MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

RUN_NAME     = "bottle_v3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
os.makedirs(RUN_ROOT_DIR, exist_ok=True)

master_csv_path       = os.path.join(BASE_OUTPUT_DIR, "all_trajectories_summary.csv")
CURRENT_TRAJECTORY_ID = 1
ACTIVE_CAMERA_INDEX   = CAMERA_INDEX

# (populated by set_trajectory_output_paths)
OUTPUT_DIR = output_video_path = framewise_csv_path = ""
boundary_csv_path = segment_csv_path = trajectory_summary_csv_path = ""
json_path = final_image_path = preview_image_path = ""


def set_trajectory_output_paths(tid):
    global CURRENT_TRAJECTORY_ID, OUTPUT_DIR
    global output_video_path, framewise_csv_path
    global boundary_csv_path, segment_csv_path
    global trajectory_summary_csv_path, json_path
    global final_image_path, preview_image_path

    CURRENT_TRAJECTORY_ID = int(tid)
    prefix     = f"trajectory_{CURRENT_TRAJECTORY_ID:03d}"
    OUTPUT_DIR = os.path.join(RUN_ROOT_DIR, prefix)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_video_path           = os.path.join(OUTPUT_DIR, f"{prefix}_video.mp4")
    framewise_csv_path          = os.path.join(OUTPUT_DIR, "framewise_data.csv")
    boundary_csv_path           = os.path.join(OUTPUT_DIR, "boundary_points.csv")
    segment_csv_path            = os.path.join(OUTPUT_DIR, "temporal_segments.csv")
    trajectory_summary_csv_path = os.path.join(OUTPUT_DIR, "trajectory_summary.csv")
    json_path                   = os.path.join(OUTPUT_DIR, "segmentation_output.json")
    final_image_path            = os.path.join(OUTPUT_DIR, f"{prefix}_graph.png")
    preview_image_path          = os.path.join(OUTPUT_DIR, f"{prefix}_graph_preview.png")


set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_finite(p):
    if p is None: return False
    try:   return bool(np.isfinite(float(p[0])) and np.isfinite(float(p[1])))
    except: return False

def is_nan(v):
    try:   return bool(np.isnan(float(v)))
    except: return True

def clip_bbox(bbox, w, h):
    x1,y1,x2,y2 = [float(v) for v in bbox]
    x1,y1 = int(max(0,min(w-1,x1))), int(max(0,min(h-1,y1)))
    x2,y2 = int(max(0,min(w-1,x2))), int(max(0,min(h-1,y2)))
    if x2<=x1: x2=min(w-1,x1+2)
    if y2<=y1: y2=min(h-1,y1+2)
    return x1,y1,x2,y2

def bbox_metrics(bbox):
    x1,y1,x2,y2 = bbox
    bw = max(1.0, float(x2-x1))
    bh = max(1.0, float(y2-y1))
    return {"bbox_width":bw,"bbox_height":bh,"bbox_area":bw*bh,"aspect_ratio":bw/bh}

def unit_vec(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9: return np.array([0.0, -1.0])
    return np.asarray(v, dtype=np.float64) / n

def segment_label(sid): return f"Segment_{sid}"


# ══════════════════════════════════════════════════════════════════════════════
#  ★  ROTATION-ROBUST OBB CAP TRACKER
#     THE KEY NEW CLASS — solves the flip problem
# ══════════════════════════════════════════════════════════════════════════════

class CapEndMemory:
    """
    Maintains a temporally-consistent estimate of the bottle's long axis
    direction such that axis[0] always points TOWARD the cap end.

    Why this solves the flip:
    ─────────────────────────
    Every frame, minAreaRect gives us a long axis, but with arbitrary sign
    (OpenCV may return it pointing either way).  We compare the new axis
    against the stored axis using a dot product:
      • dot > COS_FLIP_THRESHOLD  → same direction, accept as-is
      • dot ≤ COS_FLIP_THRESHOLD  → axis is flipped; stage a flip vote

    A flip vote is only committed after FLIP_CONFIRM_FRAMES consecutive
    frames all vote "flip".  This makes single-frame noise completely
    harmless.  A real 180° rotation takes many frames (>10 at 30fps),
    so the confirm window is safely shorter than any real motion.

    Initialisation:
    ──────────────
    First frame: geometric init — cap end = narrower cross-section.
    If cross-sections are equal: cap = the end that is higher in the image.
    After init: temporal continuity takes over.
    """

    def __init__(self):
        self.axis       = None    # unit vector pointing cap-ward
        self.cap_end    = None    # (x,y) of cap end
        self.other_end  = None    # (x,y) of base end
        self._flip_votes = 0      # consecutive frames voting for flip

    def reset(self):
        self.axis       = None
        self.cap_end    = None
        self.other_end  = None
        self._flip_votes = 0

    def update(self, end_A, end_B, mask_bin, frame_shape):
        """
        Given two candidate endpoints (end_A, end_B) of the OBB long axis,
        decide which is the cap, update internal state, return (cap, base).

        Parameters
        ----------
        end_A, end_B  : (float,float) candidate end midpoints
        mask_bin      : np.ndarray uint8 (H,W) binarised mask, or None
        frame_shape   : (H,W)

        Returns
        -------
        cap_end   : (float,float)
        other_end : (float,float)
        """
        # New candidate axis from A→B
        raw_axis_AB = np.array([end_B[0]-end_A[0], end_B[1]-end_A[1]], dtype=np.float64)
        raw_axis_BA = -raw_axis_AB

        # ── First-ever call: geometric initialisation ────────────────────────
        if self.axis is None:
            cap, base = self._geometric_init(end_A, end_B, mask_bin)
            # Set axis to point from base → cap
            self.axis      = unit_vec(np.array([cap[0]-base[0], cap[1]-base[1]]))
            self.cap_end   = cap
            self.other_end = base
            self._flip_votes = 0
            return self.cap_end, self.other_end

        # ── Subsequent calls: temporal continuity ────────────────────────────
        # Align raw_axis_AB with stored axis
        dot_AB = float(np.dot(unit_vec(raw_axis_AB), self.axis))
        dot_BA = float(np.dot(unit_vec(raw_axis_BA), self.axis))

        if dot_AB >= dot_BA:
            # end_B is in the cap direction
            candidate_cap, candidate_base = end_B, end_A
            dominant_dot = dot_AB
        else:
            # end_A is in the cap direction
            candidate_cap, candidate_base = end_A, end_B
            dominant_dot = dot_BA

        # Check for flip
        if dominant_dot < COS_FLIP_THRESHOLD:
            # Axis appears to have flipped — stage a vote
            self._flip_votes += 1
            if self._flip_votes >= FLIP_CONFIRM_FRAMES:
                # Confirmed flip: update axis & ends
                self.axis        = unit_vec(np.array([candidate_cap[0]-candidate_base[0],
                                                       candidate_cap[1]-candidate_base[1]]))
                self.cap_end     = candidate_cap
                self.other_end   = candidate_base
                self._flip_votes = 0
            else:
                # Not confirmed yet — hold previous cap position
                # (return old values; trajectory remains stable)
                pass
        else:
            # Normal frame: update smoothly, reset flip votes
            self._flip_votes = 0
            self.axis        = unit_vec(np.array([candidate_cap[0]-candidate_base[0],
                                                   candidate_cap[1]-candidate_base[1]]))
            self.cap_end     = candidate_cap
            self.other_end   = candidate_base

        return self.cap_end, self.other_end

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _geometric_init(self, end_A, end_B, mask_bin):
        """
        Pure geometry, used ONLY on the very first frame.
        Returns (cap_end, base_end).
        """
        if mask_bin is not None:
            wA = _avg_cross_section(end_A, end_B, mask_bin)
            wB = _avg_cross_section(end_B, end_A, mask_bin)
            diff = abs(wA - wB)
            if diff > 4:          # clear width difference
                return (end_A, end_B) if wA < wB else (end_B, end_A)

        # Fall back to y-position (higher in frame = cap for upright bottle)
        return (end_A, end_B) if end_A[1] <= end_B[1] else (end_B, end_A)


def _avg_cross_section(near_end, far_end, mask_bin):
    """
    Average cross-section width of mask_bin near `near_end`,
    sampled at multiple depths along the axis toward `far_end`.
    """
    if mask_bin is None:
        return 0.0
    h, w = mask_bin.shape[:2]
    ax   = far_end[0] - near_end[0]
    ay   = far_end[1] - near_end[1]
    norm = math.sqrt(ax*ax + ay*ay)
    if norm < 1e-6:
        return 0.0
    perp_x, perp_y = -ay/norm, ax/norm   # perpendicular unit vector

    total = 0.0
    for t in CROSS_SECTION_DEPTHS:
        px = near_end[0] + t * ax
        py = near_end[1] + t * ay
        count = 0
        for d in range(-70, 71):
            sx = int(round(px + d * perp_x))
            sy = int(round(py + d * perp_y))
            if 0 <= sx < w and 0 <= sy < h and mask_bin[sy, sx] > 127:
                count += 1
        total += count
    return total / len(CROSS_SECTION_DEPTHS)


# ══════════════════════════════════════════════════════════════════════════════
#  OBB COMPUTATION FROM MASK
# ══════════════════════════════════════════════════════════════════════════════

def obb_from_mask(mask_float, w_frame, h_frame):
    """
    Fit an OBB to the largest contour in the (float) segmentation mask.

    Returns
    -------
    box_pts : np.ndarray (4,2) float32   OBB corners, or None on failure
    end_A   : (float,float)  midpoint of one long-axis end
    end_B   : (float,float)  midpoint of other long-axis end
    """
    try:
        mask_bin = (mask_float > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, mask_bin

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < MIN_MASK_AREA_PX:
            return None, None, None, mask_bin

        rect    = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect).astype(np.float32)

        # Two candidate axis-end midpoints
        # minAreaRect corners: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
        # Long axis: pair (0-3) vs (1-2)  or  pair (0-1) vs (2-3)
        def mid(p, q): return ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)

        mAB = mid(box_pts[0], box_pts[3])   # edge 0-3
        mCD = mid(box_pts[1], box_pts[2])   # edge 1-2
        mEF = mid(box_pts[0], box_pts[1])   # edge 0-1
        mGH = mid(box_pts[2], box_pts[3])   # edge 2-3

        dAB_CD = euclidean(mAB, mCD)
        dEF_GH = euclidean(mEF, mGH)

        if dAB_CD >= dEF_GH:
            end_A, end_B = mAB, mCD
        else:
            end_A, end_B = mEF, mGH

        return box_pts, end_A, end_B, mask_bin

    except Exception:
        return None, None, None, None


def obb_heuristic_from_aabb(bbox):
    """
    Tracker-frame fallback: no mask available.
    Returns (end_A, end_B) from AABB geometry.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2-x1, y2-y1
    cx, cy = (x1+x2)/2, (y1+y2)/2
    if bh >= bw:
        return (cx, y1+0.08*bh), (cx, y2-0.08*bh)
    else:
        return (x1+0.08*bw, cy), (x2-0.08*bw, cy)


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION SELECTION  (nearest to last anchor)
# ══════════════════════════════════════════════════════════════════════════════

def best_detection(detections, last_anchor):
    if not detections: return None
    if last_anchor is None:
        return max(detections, key=lambda d: d["confidence"])
    return min(detections, key=lambda d: euclidean(d["anchor"], last_anchor))


# ══════════════════════════════════════════════════════════════════════════════
#  OPENCV TRACKER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_cv_tracker():
    if hasattr(cv2, "legacy"):
        for name in ("TrackerCSRT_create","TrackerKCF_create","TrackerMIL_create"):
            if hasattr(cv2.legacy, name):
                try: return getattr(cv2.legacy, name)()
                except: pass
    for name in ("TrackerCSRT_create","TrackerKCF_create","TrackerMIL_create"):
        if hasattr(cv2, name):
            try: return getattr(cv2, name)()
            except: pass
    return None

def xyxy_to_xywh(b): return (int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1]))
def xywh_to_xyxy(b): return (int(b[0]),int(b[1]),int(b[0]+b[2]),int(b[1]+b[3]))


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def savgol_smooth(points):
    if len(points) < 3 or not SCIPY_OK:
        return points[-1]
    local = points[-SAVGOL_WINDOW:] if len(points) >= SAVGOL_WINDOW else points[:]
    n   = len(local)
    win = SAVGOL_WINDOW if SAVGOL_WINDOW <= n else (n if n%2==1 else n-1)
    if win%2==0: win -= 1
    if win < 3:  return local[-1]
    poly = min(SAVGOL_POLYORDER, win-1)
    xs = np.array([p[0] for p in local], dtype=np.float32)
    ys = np.array([p[1] for p in local], dtype=np.float32)
    try:
        from scipy.signal import savgol_filter
        return float(savgol_filter(xs,win,poly,mode="interp")[-1]), \
               float(savgol_filter(ys,win,poly,mode="interp")[-1])
    except:
        return local[-1]

def ema_filter(cur, prev, alpha):
    if prev is None: return cur
    return (alpha*cur[0]+(1-alpha)*prev[0],
            alpha*cur[1]+(1-alpha)*prev[1])


# ══════════════════════════════════════════════════════════════════════════════
#  START DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_sustained_motion_start(pre_buf):
    buf = list(pre_buf)
    if len(buf) < START_LOOKBACK_FRAMES + 1:
        return None
    recent  = buf[-(START_LOOKBACK_FRAMES+1):]
    p_old   = np.array(recent[0][1],  dtype=np.float32)
    p_now   = np.array(recent[-1][1], dtype=np.float32)
    disp    = float(np.linalg.norm(p_now - p_old))
    speeds  = [euclidean(recent[i-1][1], recent[i][1]) for i in range(1,len(recent))]
    med_spd = float(np.median(speeds)) if speeds else 0.0
    if disp >= START_DISPLACEMENT_PIXELS and med_spd >= START_MIN_MEDIAN_SPEED_PIXELS:
        return [item for item in buf if item[0] >= recent[0][0]]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-SCALE DIRECTION-CHANGE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _angle_between(v1, v2):
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < MIN_VECTOR_LENGTH_PIXELS or n2 < MIN_VECTOR_LENGTH_PIXELS: return 0.0
    cos_v = float(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_v))

def _signed_heading(v1, v2):
    a1, a2 = math.atan2(float(v1[1]),float(v1[0])), math.atan2(float(v2[1]),float(v2[0]))
    d = math.degrees(a2-a1)
    while d > 180: d -= 360
    while d < -180: d += 360
    return d

def _multiscale_candidate(smooth_pts, frame_nums):
    best = None
    for w in DIRECTION_WINDOWS:
        if len(smooth_pts) < 2*w+1: continue
        po = np.array(smooth_pts[-1-2*w], dtype=np.float32)
        pm = np.array(smooth_pts[-1-w],   dtype=np.float32)
        pn = np.array(smooth_pts[-1],     dtype=np.float32)
        vb, va = pm-po, pn-pm
        ang    = _angle_between(vb, va)
        sgn    = _signed_heading(vb, va)
        c = {"frame_number": int(frame_nums[-1-w]),
             "x": float(smooth_pts[-1-w][0]),
             "y": float(smooth_pts[-1-w][1]),
             "direction_change": float(ang),
             "signed_direction_change": float(sgn),
             "direction_window": int(w)}
        if best is None or ang > best["direction_change"]:
            best = c
    return (best, best["direction_change"]) if best else (None, 0.0)


class DirectionChangeDetector:
    def __init__(self, first_frame):
        self.turn_active   = False
        self.best_cand     = None
        self.first_frame   = int(first_frame)

    def _valid(self, c, bds):
        if c is None: return False
        if MAX_BOUNDARIES is not None and len(bds) >= MAX_BOUNDARIES: return False
        prev = bds[-1]["frame_number"] if bds else self.first_frame
        if c["frame_number"] - prev < MIN_SEGMENT_LENGTH_FRAMES: return False
        if bds and c["frame_number"] - bds[-1]["frame_number"] < MIN_BOUNDARY_GAP_FRAMES:
            return False
        return True

    def _commit(self, c, bds):
        if not self._valid(c, bds): return None
        b = dict(c)
        b["boundary_id"]    = len(bds)+1
        b["cue_type"]       = "direction_change_hysteresis"
        b["boundary_score"] = c["direction_change"] / max(ANGLE_CHANGE_THRESHOLD,1e-6)
        return b

    def update(self, smooth_pts, frame_nums, bds):
        cand, val = _multiscale_candidate(smooth_pts, frame_nums)
        if cand is None: return None, 0.0
        if val >= ANGLE_CHANGE_THRESHOLD:
            self.turn_active = True
            if self.best_cand is None or val > self.best_cand["direction_change"]:
                self.best_cand = cand
            return None, val
        if self.turn_active and val <= ANGLE_RESET_THRESHOLD:
            b = self._commit(self.best_cand, bds)
            self.turn_active = False
            self.best_cand   = None
            return b, val
        return None, val

    def flush(self, bds):
        b = self._commit(self.best_cand, bds) if (self.turn_active and self.best_cand) else None
        self.turn_active = False
        self.best_cand   = None
        return b


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_seg_id(frame_no, bds):
    sid = 1
    for b in bds:
        if frame_no > b["frame_number"]: sid += 1
    return sid

def seg_for_frame(frame_no, bds, start_frame):
    if start_frame is None or frame_no < start_frame: return 0, "Before_START"
    return get_seg_id(frame_no, bds), segment_label(get_seg_id(frame_no, bds))

def make_segments(bds, first, last):
    if last <= 0 or first is None: return []
    segs, start, sid = [], int(first), 1
    for b in bds:
        bf = int(b["frame_number"])
        if bf >= start:
            segs.append({"segment_id":sid,"label":segment_label(sid),
                          "start_frame":start,"end_frame":bf,
                          "duration_frames":bf-start+1})
            start, sid = bf+1, sid+1
    if start <= last:
        segs.append({"segment_id":sid,"label":segment_label(sid),
                      "start_frame":start,"end_frame":last,
                      "duration_frames":last-start+1})
    return segs


# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def _txt(img, text, pos, scale=0.50, color=(255,255,255), thick=1):
    """Draw text with black shadow."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)

def draw_overlay(frame, state, tracking_src):
    smooth_pts = state["smooth_points"]
    bds        = state["boundaries"]
    start_pt   = state["start_point"]
    cap_end    = state["last_cap_end"]
    obb_box    = state["last_obb_box"]
    last_bbox  = state["last_bbox"]

    # OBB outline
    if obb_box is not None:
        pts_i = obb_box.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(frame, [pts_i], True, OBB_COLOR, 2, cv2.LINE_AA)

    # Cap dot (magenta)
    if cap_end is not None and is_finite(cap_end):
        cx, cy = int(cap_end[0]), int(cap_end[1])
        cv2.circle(frame, (cx,cy), 11, CAP_DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 11, (0,0,0), 1, cv2.LINE_AA)
        _txt(frame, "CAP", (cx+13, cy-12), scale=0.48, color=(255,0,255))

    # AABB fallback (faint)
    if last_bbox is not None and obb_box is None:
        x1,y1,x2,y2 = last_bbox
        cv2.rectangle(frame, (x1,y1),(x2,y2), (180,180,180), 1)

    # Trajectory line
    valid = [(int(p[0]),int(p[1])) for p in smooth_pts if is_finite(p)]
    for i in range(1, len(valid)):
        cv2.line(frame, valid[i-1], valid[i], TRJ_COLOR, 3, cv2.LINE_AA)

    # Start dot (green)
    if start_pt is not None and is_finite(start_pt):
        cv2.circle(frame, (int(start_pt[0]),int(start_pt[1])),
                   9, (0,255,0), -1, cv2.LINE_AA)
        _txt(frame, "START", (int(start_pt[0])+12, int(start_pt[1])-12),
             color=(0,230,0))

    # Boundary dots (red)
    for b in bds:
        cv2.circle(frame, (int(b["x"]),int(b["y"])), 11, (0,0,255), -1, cv2.LINE_AA)
        _txt(frame, f"S{b['boundary_id']}", (int(b["x"])+13, int(b["y"])-12),
             color=(255,80,80))

    # Current end-point (orange)
    if SHOW_CURRENT_POINT and smooth_pts:
        p = smooth_pts[-1]
        if is_finite(p):
            cv2.circle(frame, (int(p[0]),int(p[1])), 7, (0,165,255), -1, cv2.LINE_AA)

def draw_status(frame, state, fid, tracking_src):
    lines = [
        f"T{CURRENT_TRAJECTORY_ID:03d} | Frame:{fid} | "
        f"track:{tracking_src} | flip_votes:{state['cap_memory']._flip_votes}",
        f"pts={len(state['smooth_points'])}  bds={len(state['boundaries'])}",
        "[s] save+next    [r] reset    [q] save+quit",
    ]
    y = 24
    for line in lines:
        _txt(frame, line, (10, y), scale=0.50, color=(0,255,255))
        y += 22


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def render_live_graph(smooth_pts, bds, start_pt):
    fig    = Figure(figsize=(LIVE_GRAPH_WIN_W/LIVE_GRAPH_DPI,
                              LIVE_GRAPH_WIN_H/LIVE_GRAPH_DPI), dpi=LIVE_GRAPH_DPI)
    canvas = FigureCanvas(fig)
    ax     = fig.add_subplot(111)
    valid  = [p for p in smooth_pts if is_finite(p)]
    if len(valid) >= 2:
        xs, ys = [p[0] for p in valid], [p[1] for p in valid]
        ax.plot(xs, ys, color="blue", lw=2.5, label="Trajectory", zorder=3)
        ax.scatter([xs[-1]],[ys[-1]], s=MARKER_SIZE, color="orange",
                   edgecolors="black", lw=0.7, label="End point", zorder=6)
        pad = GRAPH_PADDING_PIXELS
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(max(ys)+pad, min(ys)-pad)
    if is_finite(start_pt):
        ax.scatter([float(start_pt[0])],[float(start_pt[1])], s=MARKER_SIZE,
                   color="lime", edgecolors="black", lw=0.7, label="Start", zorder=7)
    if bds:
        ax.scatter([b["x"] for b in bds],[b["y"] for b in bds],
                   s=MARKER_SIZE, color="red", edgecolors="black", lw=0.7,
                   label="Boundary", zorder=8)
    ax.grid(True, ls="--", lw=0.6, alpha=0.4, color="grey")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black",
              framealpha=1.0, fontsize=8, title="Legend", title_fontsize=9)
    fig.tight_layout(rect=[0,0.04,1,1])
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = cv2.cvtColor(buf[...,:3], cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (LIVE_GRAPH_WIN_W, LIVE_GRAPH_WIN_H),
                     interpolation=cv2.INTER_AREA)
    plt.close(fig)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SAVED GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def save_final_graph(smooth_pts, bds, start_pt):
    valid = [p for p in smooth_pts if is_finite(p)]
    if not valid:
        print("[Graph] No valid points – skip.", flush=True); return
    xs, ys = [float(p[0]) for p in valid], [float(p[1]) for p in valid]
    pad    = GRAPH_PADDING_PIXELS
    x_min, x_max = max(0,min(xs)-pad), max(xs)+pad
    y_min, y_max = max(0,min(ys)-pad), max(ys)+pad
    for mn, mx in [(x_min,x_max),(y_min,y_max)]:
        if mx - mn < 220:
            mid = 0.5*(mn+mx); mn = max(0,mid-110); mx = mid+110

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.plot(xs, ys, color="blue", lw=2.8, label="Trajectory", zorder=3)
    if is_finite(start_pt):
        ax.scatter([float(start_pt[0])],[float(start_pt[1])], s=MARKER_SIZE,
                   color="lime", edgecolors="black", lw=0.8, label="Start", zorder=6)
    if bds:
        bx, by = [b["x"] for b in bds], [b["y"] for b in bds]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red", edgecolors="black",
                   lw=0.8, label="Boundary", zorder=7)
        for b in bds:
            ax.annotate(f"S{b['boundary_id']}\nF{b['frame_number']}",
                        xy=(b["x"],b["y"]), xytext=(10,10),
                        textcoords="offset points", fontsize=8, color="red",
                        fontweight="bold",
                        bbox=dict(facecolor="white",edgecolor="none",pad=1.5))
    ax.scatter([xs[-1]],[ys[-1]], s=MARKER_SIZE, color="orange",
               edgecolors="black", lw=0.8, label="End", zorder=6)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.invert_yaxis(); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.annotate(X_LABEL_TEXT, xy=(1,0), xycoords=("axes fraction","axes fraction"),
                xytext=(6,-20), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="top", annotation_clip=False)
    ax.annotate(Y_LABEL_TEXT, xy=(0,1), xycoords=("axes fraction","axes fraction"),
                xytext=(-30,6), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="bottom", annotation_clip=False)
    ax.grid(True, ls="--", lw=0.8, alpha=0.45, color="grey")
    for tick in ax.get_xticklabels()+ax.get_yticklabels():
        tick.set_fontsize(TICK_LABEL_FONTSIZE); tick.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", length=6, width=1.4, direction="out")
    for sp in ax.spines.values(): sp.set_linewidth(1.3); sp.set_color("black")
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0,
                    fontsize=LEGEND_FONTSIZE, title="Legend",
                    title_fontsize=LEGEND_TITLE_FONTSIZE,
                    borderpad=0.8, labelspacing=0.7)
    leg.get_title().set_fontweight("bold"); leg.get_frame().set_linewidth(1.1)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.92)
    fig.text(0.02,0.02,FOOTER_TEXT,fontsize=8,color="black",
             wrap=True,ha="left",va="bottom")
    fig.savefig(final_image_path,   dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(preview_image_path, dpi=150,      bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Graph] Saved → {final_image_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV / JSON SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_framewise_csv(records, bds, start_frame):
    bd_map = {int(b["frame_number"]):b for b in bds}
    with open(framewise_csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_number","segment_id","segment_label",
                    "tracking_source","detected","confidence",
                    "x_raw","y_raw","x_smooth","y_smooth",
                    "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                    "bbox_width","bbox_height","bbox_area","aspect_ratio",
                    "anchor_mode","boundary_id","boundary_cue","is_boundary",
                    "motion_start_frame"])
        for r in records:
            fid = int(r["frame_number"])
            sid, slbl = seg_for_frame(fid, bds, start_frame)
            b = bd_map.get(fid)
            w.writerow([fid,sid,slbl,
                        r["tracking_source"],r["detected"],r["confidence"],
                        r["x_raw"],r["y_raw"],r["x_smooth"],r["y_smooth"],
                        r["bbox_x1"],r["bbox_y1"],r["bbox_x2"],r["bbox_y2"],
                        r["bbox_width"],r["bbox_height"],r["bbox_area"],r["aspect_ratio"],
                        "obb_cap_temporal",
                        b["boundary_id"] if b else "",
                        b["cue_type"]    if b else "none",
                        bool(b),
                        start_frame if start_frame is not None else ""])

def save_boundary_csv(bds):
    with open(boundary_csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary_id","frame_number","x","y",
                    "direction_change_deg","signed_direction_change_deg",
                    "direction_window","boundary_score","cue_type"])
        for b in bds:
            w.writerow([b["boundary_id"],b["frame_number"],b["x"],b["y"],
                        b.get("direction_change",""),
                        b.get("signed_direction_change",""),
                        b.get("direction_window",""),
                        b.get("boundary_score",""),
                        b.get("cue_type","")])

def save_segment_csv(segs):
    with open(segment_csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id","label","start_frame","end_frame","duration_frames"])
        for s in segs:
            w.writerow([s["segment_id"],s["label"],
                        s["start_frame"],s["end_frame"],s["duration_frames"]])

def save_summary_csv(records, bds, segs, start_frame, start_pt, total):
    if not records: return
    valid = [r for r in records if not is_nan(r["x_smooth"]) and not is_nan(r["y_smooth"])]
    if not valid: return
    last = valid[-1]
    tid  = f"{RUN_NAME}_trajectory_{CURRENT_TRAJECTORY_ID:03d}"
    row  = {
        "trajectory_id":     tid,
        "trajectory_number": CURRENT_TRAJECTORY_ID,
        "target_class":      TARGET_CLASS,
        "camera_index":      ACTIVE_CAMERA_INDEX,
        "anchor_mode":       "obb_cap_temporal",
        "total_frames":      total,
        "start_frame":       start_frame if start_frame is not None else "",
        "start_x":           float(start_pt[0]) if is_finite(start_pt) else "",
        "start_y":           float(start_pt[1]) if is_finite(start_pt) else "",
        "end_frame":         int(last["frame_number"]),
        "end_x":             float(last["x_smooth"]),
        "end_y":             float(last["y_smooth"]),
        "num_boundaries":    len(bds),
        "num_segments":      len(segs),
        "boundary_frames":   ";".join(str(b["frame_number"]) for b in bds),
        "segment_ranges":    ";".join(f"{s['label']}:{s['start_frame']}-{s['end_frame']}"
                                      for s in segs),
        "output_folder":     OUTPUT_DIR,
        "final_graph":       final_image_path,
    }
    with open(trajectory_summary_csv_path,"w",newline="") as f:
        dw = csv.DictWriter(f, fieldnames=list(row.keys()))
        dw.writeheader(); dw.writerow(row)
    master_exists = os.path.exists(master_csv_path)
    with open(master_csv_path,"a",newline="") as f:
        dw = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not master_exists: dw.writeheader()
        dw.writerow(row)
    print(f"[T{CURRENT_TRAJECTORY_ID:03d}] pts={len(valid)}  "
          f"bds={len(bds)}  segs={len(segs)}", flush=True)

def save_json(start_frame, start_pt, bds, segs, total):
    with open(json_path,"w") as f:
        json.dump({"trajectory_id":CURRENT_TRAJECTORY_ID,
                   "camera_index":ACTIVE_CAMERA_INDEX,
                   "target_class":TARGET_CLASS,
                   "anchor_mode":"obb_cap_temporal",
                   "total_frames":total,
                   "motion_start":({"frame_number":int(start_frame),
                                    "x":float(start_pt[0]),
                                    "y":float(start_pt[1])}
                                   if start_frame is not None and is_finite(start_pt)
                                   else None),
                   "boundaries":bds, "segments":segs,
                   "output_files":{"framewise_csv":framewise_csv_path,
                                   "boundary_csv":boundary_csv_path,
                                   "segment_csv":segment_csv_path,
                                   "summary_csv":trajectory_summary_csv_path,
                                   "final_graph":final_image_path,
                                   "preview_graph":preview_image_path,
                                   "video":output_video_path}},
                  f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE COMPLETE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_trajectory(state):
    if not state["records"]:
        print("[Save] Nothing to save.", flush=True); return
    total  = int(state["frame_number"])
    sf     = state["start_frame"]
    sp     = state["start_point"]
    final_b = list(state["boundaries"])
    final_s = make_segments(final_b, sf, total)
    save_framewise_csv(state["records"], final_b, sf)
    save_boundary_csv(final_b)
    save_segment_csv(final_s)
    save_summary_csv(state["records"], final_b, final_s, sf, sp, total)
    save_json(sf, sp, final_b, final_s, total)
    save_final_graph(state["smooth_points"], final_b, sp)
    if state["video_writer"] is not None:
        state["video_writer"].release()
        state["video_writer"] = None
    print(f"[Save] T{CURRENT_TRAJECTORY_ID} → {OUTPUT_DIR}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def fresh_state():
    return {
        "frame_number":      0,
        "records":           [],
        "raw_points":        [],
        "smooth_points":     [],
        "frame_numbers":     [],
        "prev_ema":          None,
        # Tracking
        "last_anchor":       None,
        "last_bbox":         None,
        "last_metrics":      None,
        "last_obb_box":      None,
        "last_cap_end":      None,
        # ★ Core new component
        "cap_memory":        CapEndMemory(),
        # CV tracker
        "tracker":           None,
        "tracker_only_count":0,
        # Start detection
        "pre_start_buf":     deque(maxlen=PRE_START_BUFFER_FRAMES),
        "movement_started":  False,
        "start_frame":       None,
        "start_point":       None,
        # Segmentation
        "dir_detector":      None,
        "boundaries":        [],
        # Output
        "video_writer":      None,
        "has_seg_model":     False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CURRENT_TRAJECTORY_ID

    # ── Load model ────────────────────────────────────────────────────────────
    model, has_seg_model, loaded_path = None, False, None
    for cand in MODEL_PATH_CANDIDATES:
        exists = os.path.exists(cand) or cand in ("yolov8n-seg.pt","yolov8n.pt")
        if not exists: continue
        print(f"[YOLO] Trying: {cand}", flush=True)
        try:
            model         = YOLO(cand)
            loaded_path   = cand
            has_seg_model = "seg" in cand
            break
        except Exception as e:
            print(f"[YOLO] Failed: {e}", flush=True)

    if model is None:
        print("[CRITICAL] Could not load YOLO model.", flush=True); return

    if not has_seg_model:
        print("[WARN] Detection model loaded (no instance mask).\n"
              "       OBB anchor will use AABB heuristic – accuracy reduced.\n"
              "       Place yolov8n-seg.pt in yolo-Weights/ for full robustness.",
              flush=True)
    else:
        print("[YOLO] Seg model active. Temporal OBB cap-end tracking ENABLED.",
              flush=True)

    target_cls_id = next((cid for cid,cname in model.names.items()
                          if cname == TARGET_CLASS), None)
    if target_cls_id is None:
        print(f"[CRITICAL] '{TARGET_CLASS}' not found in model.", flush=True); return

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(ACTIVE_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[CRITICAL] Camera {ACTIVE_CAMERA_INDEX} unavailable.", flush=True)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    ret, init_f = cap.read()
    if not ret:
        print("[CRITICAL] Empty first frame.", flush=True); cap.release(); return
    h, w = init_f.shape[:2]
    print(f"[Camera] {w}×{h} @ {CAMERA_FPS}fps  |  "
          f"Target: '{TARGET_CLASS}'  |  Conf: {CONF_THRESHOLD}", flush=True)
    print("\n>>> [s] save+next   [r] reset   [q] save+quit\n", flush=True)

    state = fresh_state()
    state["has_seg_model"] = has_seg_model
    last_graph_img = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Loop] Frame grab failure.", flush=True); break

            state["frame_number"] += 1
            fid = state["frame_number"]

            # Lazy video-writer init
            if SAVE_ANNOTATED_VIDEO and state["video_writer"] is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                state["video_writer"] = cv2.VideoWriter(
                    output_video_path, fourcc, CAMERA_FPS, (w, h))

            # ══════════════════════════════════════════════════════════════════
            #  DETECTION PASS
            # ══════════════════════════════════════════════════════════════════
            selected       = None
            tracking_src   = "missing"
            raw_anchor     = None
            confidence     = 0.0

            results = model.predict(frame, conf=CONF_THRESHOLD,
                        imgsz=YOLO_IMGSZ, verbose=False)

            detections = []
            boxes = results[0].boxes

            # print("Detections:", len(boxes))

            masks_obj = (
                results[0].masks
                if has_seg_model and results[0].masks is not None
                else None
            )

            for idx, box in enumerate(boxes):
                if int(box.cls[0]) != target_cls_id:
                    continue
                bbox = clip_bbox(box.xyxy[0].cpu().numpy(), w, h)
                conf = float(box.conf[0])

                # Extract + resize mask
                mask_arr = None
                if masks_obj is not None and idx < len(masks_obj.data):
                    m = masks_obj.data[idx].cpu().numpy().astype(np.float32)
                    mask_arr = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

                # ── Compute OBB & temporal cap end ────────────────────────────
                if mask_arr is not None:
                    obb_box, end_A, end_B, mask_bin = obb_from_mask(mask_arr, w, h)
                else:
                    obb_box = None
                    end_A, end_B = obb_heuristic_from_aabb(bbox)
                    mask_bin     = None

                if end_A is not None and end_B is not None:
                    # Temporal cap selection (uses CapEndMemory per-detection;
                    # state.cap_memory is the primary persistent one)
                    cap_end, _ = state["cap_memory"].update(
                        end_A, end_B, mask_bin, (h, w)
                    )

                    if state["last_cap_end"] is not None:
                        jump = euclidean(cap_end, state["last_cap_end"])

                        # if jump > 120:
                        #     cap_end = state["last_cap_end"]

                    anchor = cap_end
                else:
                    # Fallback: AABB upper centre
                    x1,y1,x2,y2 = [float(v) for v in bbox]
                    anchor = ((x1+x2)/2, y1 + 0.20*(y2-y1))
                    obb_box = None
                    cap_end = anchor

                detections.append({
                    "bbox":       bbox,
                    "anchor":     anchor,
                    "confidence": conf,
                    "obb_box":    obb_box,
                    "cap_end":    cap_end,
                })

            # ── Select best detection ─────────────────────────────────────────
            chosen = best_detection(detections, state["last_anchor"])

            if chosen is not None:
                print("Bottle lost at frame", fid)
                state["last_bbox"]    = chosen["bbox"]
                state["last_anchor"]  = chosen["anchor"]
                state["last_metrics"] = bbox_metrics(chosen["bbox"])
                state["last_obb_box"] = chosen["obb_box"]
                state["last_cap_end"] = chosen["cap_end"]
                state["tracker_only_count"] = 0
                raw_anchor = chosen["anchor"]
                confidence = chosen["confidence"]

                if USE_CV_TRACKER:
                    state["tracker"] = create_cv_tracker()
                    if state["tracker"] is not None:
                        try:
                            state["tracker"].init(frame, xyxy_to_xywh(chosen["bbox"]))
                        except:
                            state["tracker"] = None

            elif (USE_CV_TRACKER and state["tracker"] is not None
                  and state["tracker_only_count"] < MAX_TRACKER_FRAMES):
                # ── OpenCV tracker fallback ───────────────────────────────────
                try:
                    ok, txywh = state["tracker"].update(frame)
                except:
                    ok = False

                if ok:
                    bbox = clip_bbox(xywh_to_xyxy(txywh), w, h)
                    # No mask → AABB heuristic end points
                    end_A, end_B = obb_heuristic_from_aabb(bbox)
                    # Preserve cap direction from memory (pass mask_bin=None)
                    cap_end, _ = state["cap_memory"].update(end_A, end_B, None, (h,w))

                    if state["last_cap_end"] is not None:
                        jump = euclidean(cap_end, state["last_cap_end"])

                        if jump > 40:
                            cap_end = state["last_cap_end"]
                    state["last_bbox"]    = bbox
                    state["last_anchor"]  = cap_end
                    state["last_metrics"] = bbox_metrics(bbox)
                    state["last_obb_box"] = None
                    state["last_cap_end"] = cap_end
                    state["tracker_only_count"] += 1
                    tracking_src = "tracker"
                    raw_anchor   = cap_end
                    confidence   = 0.0
                else:
                    state["tracker"] = None

            # ══════════════════════════════════════════════════════════════════
            #  POINT ACCUMULATION → SMOOTHING → START DETECTION
            # ══════════════════════════════════════════════════════════════════
            sx = sy = float("nan")

            if raw_anchor is not None:
                rp = (float(raw_anchor[0]), float(raw_anchor[1]))
                state["pre_start_buf"].append((fid, rp))

                if not state["movement_started"]:
                    seed = find_sustained_motion_start(state["pre_start_buf"])
                    if seed is not None:
                        state["movement_started"] = True
                        state["start_frame"]      = int(seed[0][0])
                        print(f"[START] Frame {state['start_frame']}", flush=True)
                        for s_fno, s_pt in seed:
                            state["raw_points"].append(s_pt)
                            sg = savgol_smooth(state["raw_points"])
                            ep = ema_filter(sg, state["prev_ema"], EMA_ALPHA)
                            state["prev_ema"] = ep
                            state["smooth_points"].append(ep)
                            state["frame_numbers"].append(int(s_fno))
                        state["start_point"]  = state["smooth_points"][0]
                        state["dir_detector"] = DirectionChangeDetector(
                            state["start_frame"])
                        sx, sy = state["smooth_points"][-1]
                else:
                    state["raw_points"].append(rp)
                    sg = savgol_smooth(state["raw_points"])
                    ep = ema_filter(sg, state["prev_ema"], EMA_ALPHA)
                    state["prev_ema"] = ep
                    state["smooth_points"].append(ep)
                    state["frame_numbers"].append(fid)
                    sx, sy = ep

            # ── Direction-change detection ─────────────────────────────────────
            if (state["movement_started"] and state["dir_detector"] is not None
                    and raw_anchor is not None):
                new_b, _ = state["dir_detector"].update(
                    state["smooth_points"], state["frame_numbers"], state["boundaries"]
                )
                if new_b is not None:
                    state["boundaries"].append(new_b)
                    print(f"  Boundary S{new_b['boundary_id']} "
                          f"F{new_b['frame_number']} "
                          f"angle={new_b['direction_change']:.1f}°", flush=True)

            cur_seg = get_seg_id(fid, state["boundaries"])

            # ── Record ────────────────────────────────────────────────────────
            m  = (state["last_metrics"] or
                  {"bbox_width":"","bbox_height":"","bbox_area":"","aspect_ratio":""})
            bc = state["last_bbox"] if state["last_bbox"] is not None else ["","","",""]
            state["records"].append({
                "frame_number":    fid,
                "tracking_source": tracking_src,
                "detected":        raw_anchor is not None,
                "confidence":      confidence,
                "x_raw":     float(raw_anchor[0]) if raw_anchor else float("nan"),
                "y_raw":     float(raw_anchor[1]) if raw_anchor else float("nan"),
                "x_smooth":  sx, "y_smooth": sy,
                "bbox_x1":bc[0],"bbox_y1":bc[1],"bbox_x2":bc[2],"bbox_y2":bc[3],
                **m,
            })

            # ── Draw & display ────────────────────────────────────────────────
            annotated = frame.copy()
            draw_overlay(annotated, state, tracking_src)
            draw_status(annotated, state, fid, tracking_src)

            if state["video_writer"] is not None:
                state["video_writer"].write(annotated)

            if SHOW_WINDOWS:
                cv2.imshow("Bottle Cap Tracker v3", annotated)
                if (SHOW_LIVE_GRAPH_WINDOW
                        and (last_graph_img is None
                             or fid % LIVE_GRAPH_UPDATE_EVERY_N == 0)):
                    last_graph_img = render_live_graph(
                        state["smooth_points"], state["boundaries"],
                        state["start_point"]
                    )
                if last_graph_img is not None:
                    cv2.imshow("Trajectory Graph", last_graph_img)

            # ── Keyboard controls ─────────────────────────────────────────────
            key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF

            if key == ord("q"):
                print("\n[Control] Quit → saving...", flush=True)
                if state["dir_detector"]:
                    fb = state["dir_detector"].flush(state["boundaries"])
                    if fb: state["boundaries"].append(fb)
                save_trajectory(state)
                break

            elif key == ord("s"):
                print(f"\n[Control] Save T{CURRENT_TRAJECTORY_ID} → next...",
                      flush=True)
                if state["dir_detector"]:
                    fb = state["dir_detector"].flush(state["boundaries"])
                    if fb: state["boundaries"].append(fb)
                save_trajectory(state)
                CURRENT_TRAJECTORY_ID += 1
                set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)
                state = fresh_state()
                state["has_seg_model"] = has_seg_model
                last_graph_img = None

            elif key == ord("r"):
                print(f"\n[Control] Reset T{CURRENT_TRAJECTORY_ID}.", flush=True)
                if state["video_writer"]: state["video_writer"].release()
                state = fresh_state()
                state["has_seg_model"] = has_seg_model
                last_graph_img = None

    except KeyboardInterrupt:
        print("\n[Control] Interrupted → saving...", flush=True)
        if state.get("dir_detector"):
            fb = state["dir_detector"].flush(state["boundaries"])
            if fb: state["boundaries"].append(fb)
        save_trajectory(state)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSCRIPT TERMINATED CLEANLY.", flush=True)
        bds = state.get("boundaries", [])
        print("\nFINAL BOUNDARIES:")
        if not bds:
            print("  None. Try lowering ANGLE_CHANGE_THRESHOLD (28→22).")
        for b in bds:
            print(f"  S{b['boundary_id']}: F{b['frame_number']} | "
                  f"({b['x']:.1f},{b['y']:.1f}) | "
                  f"angle={b['direction_change']:.1f}°")


if __name__ == "__main__":
    main()