"""
VISUAL INSPECTION & DEFECT ROOT-CAUSE ASSISTANT
NEURAX Hackathon 3.0 - Domain 2: AI in Industry and Automation

What this script does (all advisory / simulated - no hardware access):
  1. Classifies each unit with YOLO (normal vs defect classes)
  2. Calibrates confidence (temperature scaling) and flags uncertain / novel defects
  3. Cost-based accept / review / reject decision (false-accept vs false-reject handling)
  4. Localizes defects (occlusion-sensitivity heatmap + bounding box)
  5. Root-cause zone attribution (defect prior x production stress, incl. Storage)
  6. Bottleneck detection, throughput loss, WIP and batch-drift check
  7. Profitability / margin estimate, what-if scenarios, margin prediction model
  8. Evidence-based process recommendations
  9. Model evaluation: accuracy, calibration, robustness to lighting / rotation /
     blur / noise, and out-of-distribution probes
 10. Batch mode + self-contained HTML dashboard (continuously updated view)

Run:
    python inspection_system.py                    (interactive menu)
    python inspection_system.py --mode single --product 890
    python inspection_system.py --mode batch --n 60
    python inspection_system.py --mode eval
    python inspection_system.py --mode demo        (eval + batch + dashboard)
"""

import os
import io
import sys
import json
import math
import base64
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def P(path):
    """Make Windows-style paths work on any OS."""
    return path.replace("\\", os.sep)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CSV_PATH = "Model_1.csv"
MODEL2_PATH = "Model_2.csv"
MODEL_PATH = P(r"runs\classify\train\weights\best.pt")
TEST_DIR = P(r"dataset_small_final\test")
VAL_DIR = P(r"dataset_small_final\val")        # optional; used for calibration if present

OUT_DIR = "outputs"
OUTPUT_FILE = "final_inspection_result.csv"
BATCH_OUTPUT_FILE = os.path.join(OUT_DIR, "batch_inspection_results.csv")
DASHBOARD_FILE = os.path.join(OUT_DIR, "dashboard.html")
CALIBRATION_FILE = os.path.join(OUT_DIR, "calibration.json")
EVAL_FILE = os.path.join(OUT_DIR, "evaluation_report.json")
LOCALIZATION_DIR = os.path.join(OUT_DIR, "localization")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---- Decision logic (false-accept / false-reject handling) ----
CONF_THRESHOLD = 0.70          # top-1 confidence below this = uncertain
MARGIN_THRESHOLD = 0.20        # top-1 minus top-2 probability below this = uncertain
ENTROPY_THRESHOLD = 0.60       # normalized entropy (0..1) above this = uncertain
TYPE_CONF_THRESHOLD = 0.60     # share of the top defect type among defect classes
COST_FALSE_REJECT = 1.0        # relative cost of scrapping / re-checking a good part
COST_FALSE_ACCEPT = 5.0        # relative cost of shipping a defective part (escape)
AUTO_REJECT_THRESHOLD = 0.50   # P(defect) at or above this = automatic reject
# Below T_LOW the part passes; between T_LOW and AUTO_REJECT goes to manual review.
T_LOW = min(COST_FALSE_REJECT / (COST_FALSE_REJECT + COST_FALSE_ACCEPT), AUTO_REJECT_THRESHOLD)

# ---- Root-cause model ----
STRESS_THRESHOLD = 1.0         # minimum production stress before claiming a zone
K = 1.0                        # weight in defect prior x exp(K x stress)

# Heuristic prototype priors (NOT learned causal probabilities).
# Storage covers dwell time / work-in-process between stages (Model_2 data).
DEFECT_PRIORS = {
    "crack":   {"Drilling": 0.35, "Milling": 0.20, "Assembly": 0.35, "Storage": 0.10},
    "scratch": {"Drilling": 0.10, "Milling": 0.60, "Assembly": 0.20, "Storage": 0.10},
    "hole":    {"Drilling": 0.65, "Milling": 0.15, "Assembly": 0.10, "Storage": 0.10},
    "rust":    {"Drilling": 0.05, "Milling": 0.10, "Assembly": 0.10, "Storage": 0.75},
}

# ---- Localization (occlusion sensitivity) ----
OCC_GRID = 5                   # patch size = image size / OCC_GRID
OCC_STRIDE = 0.5               # stride as a fraction of the patch size
HEAT_MIN_SUPPORT = 0.50        # minimum log-odds drop to claim a localization
HEAT_BOX_THRESHOLD = 0.50      # box covers heat >= this fraction of the maximum
HEAT_DIFFUSE_AREA = 0.60       # box larger than this share of the image = "diffuse"

# ---- Evaluation / robustness ----
MAX_EVAL_IMAGES = 300
TEMPERATURE_GRID = np.arange(1.0, 5.01, 0.1)   # only softening (T>=1) - never over-confident

# ---- Batch-to-batch drift ----
DRIFT_WINDOW = 50
DRIFT_Z_LIMIT = 3.0

# ---- Economics: ASSUMED values. Replace with organizer-provided economic data. ----
ECON = {
    "price": 250.0,            # selling price per good part
    "material": 90.0,          # material cost per part started
    "processing": 60.0,        # processing cost per part started
    "rework_cost": 25.0,       # cost to rework one part
    "holding_cost": 0.5,       # cost per part held in storage / WIP in the period
}
DEFECT_RATE_ASSUMED = 0.06     # the test set is class-balanced, so it cannot give this
REWORKABLE_SHARE = {"scratch": 0.7, "rust": 0.5, "hole": 0.1, "crack": 0.0}
DEFAULT_REWORKABLE = 0.3
AVG_REWORKABLE = float(np.mean(list(REWORKABLE_SHARE.values())))
WHATIF_DEFECT_REDUCTION = 0.50 # scenario A: defects reduced by 50%
WHATIF_CAPACITY_GAIN = 0.10    # scenario B: +10% capacity at the bottleneck zone

ZONES = ["Drilling", "Milling", "Assembly"]
DECISION_TEXT = {
    "PASS": "PASS",
    "REJECT": "REJECT (defect)",
    "REVIEW": "MANUAL REVIEW",
    "REJECT_NOVEL": "REJECT (unknown / novel defect type)",
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def find_column(dataframe, possible_names):
    """First matching column; case-insensitive, ignores spaces and underscores."""
    def key(s):
        return str(s).strip().lower().replace(" ", "").replace("_", "")
    lookup = {key(c): c for c in dataframe.columns}
    for name in possible_names:
        if key(name) in lookup:
            return lookup[key(name)]
    return None


def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def read_image(path):
    """Robust image read (works with non-ASCII Windows paths)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def calc_z(value, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    return obj


def list_labeled_images(root):
    """[(path, label)] where label is the parent folder name (crack, hole, ...)."""
    items = []
    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                items.append((os.path.join(folder, f), os.path.basename(folder).lower()))
    return sorted(items)


def banner(title, width=65):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


# ============================================================
# 3. LOAD PRODUCTION DATA
# ============================================================

def load_production_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}. Put Model_1.csv in the same folder as this script.")
    m1 = pd.read_csv(CSV_PATH).dropna(axis=1, how="all").reset_index(drop=True)
    m1["Batch_ID"] = [f"BATCH_{i:04d}" for i in range(len(m1))]

    m2 = None
    for candidate in [MODEL2_PATH, "Model_2(1).csv", os.path.join(OUT_DIR, "cleaned_model2.csv")]:
        if os.path.exists(candidate):
            m2 = pd.read_csv(candidate).dropna(axis=1, how="all").reset_index(drop=True)
            print("Model 2 loaded from:", candidate)
            break
    print("Model 1 rows:", len(m1), "| Model 2 rows:", 0 if m2 is None else len(m2))
    return m1, m2


# ============================================================
# 4. BOTTLENECK + PRODUCTION IMPACT (Model 2)
# ============================================================

def analyse_production(m2, verbose=True):
    d = m2.copy()
    col = {}
    for zone in ZONES:
        col[zone + "_util"] = find_column(
            d, [f"{zone} Utilization", f"{zone.lower()}_util", f"{zone} Util"])
        col[zone + "_wait"] = find_column(
            d, [f"{zone} Queue Time", f"{zone.lower()}_wait", f"{zone} Waiting Time", f"{zone} Wait"])
    missing = [k for k, v in col.items() if v is None]
    if missing:
        raise KeyError(f"Model 2 columns not found for {missing}. Available: {list(d.columns)}")

    in_col = find_column(d, ["Entities In Part 1", "entities_in_part1", "entities_in_drilling",
                             "Entities In Drilling", "entities_in"])
    out_col = find_column(d, ["Entities Out", "entities_out", "output_parts", "parts_out"])
    if in_col is None or out_col is None:
        raise KeyError(f"Model 2 input/output columns not found. Available: {list(d.columns)}")

    stor = {
        "s1t": find_column(d, ["Part 1 Storage Time"]),
        "s2t": find_column(d, ["Part 2 Storage Time"]),
        "s1n": find_column(d, ["Part 1 Stored"]),
        "s2n": find_column(d, ["Part 2 Stored"]),
    }
    numeric_cols = list(col.values()) + [in_col, out_col] + [v for v in stor.values() if v]
    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["WIP_Stored"] = 0.0
    for k in ("s1n", "s2n"):
        if stor[k]:
            d["WIP_Stored"] += d[stor[k]].fillna(0)

    # utilization in %
    for zone in ZONES:
        u = d[col[zone + "_util"]]
        d[zone + "_Util_Pct"] = u * 100 if u.max() <= 1 else u

    # bottleneck score = normalized utilization + normalized waiting
    for zone in ZONES:
        w = d[col[zone + "_wait"]]
        span = w.max() - w.min()
        wn = (w - w.min()) / span if span > 0 else pd.Series(0.0, index=d.index)
        d[zone + "_Score"] = d[zone + "_Util_Pct"] / 100.0 + wn
    score = d[[z + "_Score" for z in ZONES]]
    d["Bottleneck_Zone"] = score.idxmax(axis=1).str.replace("_Score", "", regex=False)
    d["Bottleneck_Score"] = score.max(axis=1)

    # throughput loss
    d["Throughput_Gap"] = d[in_col] - d[out_col]
    d["Throughput_Loss_Percent"] = d["Throughput_Gap"] / d[in_col].replace(0, np.nan) * 100

    # zone summary + ranking
    summary_rows, ranking_rows = [], []
    for zone in ZONES:
        util = d[zone + "_Util_Pct"]
        wait = d[col[zone + "_wait"]]
        summary_rows.append({
            "Zone": zone,
            "Average_Utilization_Percent": util.mean(),
            "Average_Waiting_Time": wait.mean(),
            "Runs_Above_90_Percent": int((util >= 90).sum()),
            "Runs_Above_95_Percent": int((util >= 95).sum()),
        })
        runs = int((d["Bottleneck_Zone"] == zone).sum())
        ranking_rows.append({
            "Zone": zone, "Bottleneck_Runs": runs,
            "Bottleneck_Percentage": runs / len(d) * 100,
            "Average_Bottleneck_Score": d[zone + "_Score"].mean(),
        })
    zone_summary = pd.DataFrame(summary_rows)
    ranking = (pd.DataFrame(ranking_rows)
               .sort_values(["Bottleneck_Runs", "Average_Bottleneck_Score"], ascending=False)
               .reset_index(drop=True))
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))

    # utilization -> waiting table and utilization -> throughput-loss curves
    bins = [0, 50, 60, 70, 80, 90, 95, 100.0001]
    labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%"]
    uw_rows, loss_curves = [], {}
    for zone in ZONES:
        cut = pd.cut(d[zone + "_Util_Pct"], bins=bins, labels=labels, include_lowest=True)
        g = d.groupby(cut, observed=False)[col[zone + "_wait"]].mean()
        for lab in labels:
            uw_rows.append({"Zone": zone, "Utilization_Range": lab,
                            "Average_Waiting_Time": g.get(lab, np.nan)})
        cut2 = pd.cut(d[zone + "_Util_Pct"], bins=bins, include_lowest=True)
        g2 = d.groupby(cut2, observed=True)["Throughput_Loss_Percent"].mean().dropna()
        centers = np.array([iv.mid for iv in g2.index], dtype=float)
        loss_curves[zone] = (centers, g2.values.astype(float))
    util_wait = pd.DataFrame(uw_rows)

    primary, secondary = ranking.iloc[0]["Zone"], ranking.iloc[1]["Zone"]
    avg_in, avg_out = d[in_col].mean(), d[out_col].mean()
    prod = {
        "model2": d, "cols": col, "in_col": in_col, "out_col": out_col, "stor": stor,
        "zone_summary": zone_summary, "ranking": ranking, "util_wait": util_wait,
        "loss_curves": loss_curves, "primary": primary, "secondary": secondary,
        "avg_in": avg_in, "avg_out": avg_out,
        "avg_gap": d["Throughput_Gap"].mean(),
        "avg_loss_pct": d["Throughput_Loss_Percent"].mean(),
        "avg_wip": d["WIP_Stored"].mean(),
        "high_util": {z: int((d[z + "_Util_Pct"] >= 90).sum()) for z in ZONES},
    }

    folder = os.path.join(OUT_DIR, "integrated_analysis")
    os.makedirs(folder, exist_ok=True)
    zone_summary.to_csv(os.path.join(folder, "zone_performance.csv"), index=False)
    ranking.to_csv(os.path.join(folder, "bottleneck_ranking.csv"), index=False)
    util_wait.to_csv(os.path.join(folder, "utilization_waiting_analysis.csv"), index=False)
    d.to_csv(os.path.join(folder, "production_impact_results.csv"), index=False)

    if verbose:
        banner("BOTTLENECK + PRODUCTION IMPACT")
        print(ranking.to_string(index=False, formatters={
            "Bottleneck_Percentage": "{:.2f}%".format, "Average_Bottleneck_Score": "{:.3f}".format}))
        print("\nPrimary bottleneck   :", primary)
        print("Secondary constraint :", secondary)
        print("\nZone performance:")
        print(zone_summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"\nAverage input parts    : {avg_in:.1f}")
        print(f"Average output parts   : {avg_out:.1f}")
        print(f"Average throughput gap : {prod['avg_gap']:.1f} parts "
              f"({prod['avg_loss_pct']:.2f}% loss)")
        print(f"Average parts in storage (WIP): {prod['avg_wip']:.1f}")
        print("Runs with utilization >= 90%:", prod["high_util"])
    return prod


# ============================================================
# 5. BASELINES + BATCH DRIFT
# ============================================================

def build_baselines(m1, m2, prod):
    needed = ["Drilling Util", "Milling Util", "Assembly Util",
              "Drilling Waiting Time", "Milling Waiting Time", "Assembly Waiting Time"]
    for c in needed:
        if c not in m1.columns:
            raise KeyError(f"Required Model 1 column missing: {c}")
    b = {}
    for zone in ZONES:
        b[zone + " Util"] = (m1[zone + " Util"].mean(), m1[zone + " Util"].std())
        w = np.log1p(m1[zone + " Waiting Time"])
        b[zone + " Wait"] = (w.mean(), w.std())
    storage_cols = {}
    if m2 is not None:
        for key, name in (("s1t", "Part 1 Storage Time"), ("s2t", "Part 2 Storage Time"),
                          ("s1n", "Part 1 Stored"), ("s2n", "Part 2 Stored")):
            c = prod["stor"].get(key)
            if c:
                v = np.log1p(pd.to_numeric(m2[c], errors="coerce").clip(lower=0))
                b["Storage " + name] = (v.mean(), v.std())
                storage_cols["Storage " + name] = c
    return b, storage_cols


def drift_check(m1, idx):
    """Compares the last DRIFT_WINDOW rows with the whole history (assumes rows are in time order)."""
    lo = max(0, idx - DRIFT_WINDOW + 1)
    n = idx - lo + 1
    flags = []
    if n < 10:
        return flags
    for zone in ZONES:
        for label, series in (("utilization", m1[zone + " Util"]),
                              ("waiting time", np.log1p(m1[zone + " Waiting Time"]))):
            se = series.std() / math.sqrt(n)
            zval = (series.iloc[lo:idx + 1].mean() - series.mean()) / se if se > 0 else 0.0
            if abs(zval) >= DRIFT_Z_LIMIT:
                flags.append(f"{zone} {label} drift {zval:+.1f} SE over last {n} batches")
    return flags


# ============================================================
# 6. VISION: probabilities, calibration, uncertainty, decision
# ============================================================

def apply_temperature(p, T):
    """Temperature scaling on log-probabilities (identical to scaling the logits)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    lp = np.log(p) / T
    lp = lp - lp.max(axis=-1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(axis=-1, keepdims=True)


class Vision:
    def __init__(self, model_path, temperature=1.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        names = self.model.names
        self.classes = [str(names[i]).lower() for i in sorted(names)]
        self.T = float(temperature)

    def raw_probs(self, sources, batch=32):
        out = []
        for i in range(0, len(sources), batch):
            results = self.model.predict(source=sources[i:i + batch], verbose=False)
            for r in results:
                out.append(to_np(r.probs.data).astype(float))
        return np.vstack(out) if out else np.zeros((0, len(self.classes)))

    def calibrated(self, p):
        return apply_temperature(p, self.T)


def analyse_probs(p, classes):
    """Uncertainty measures + three-way decision from a (calibrated) probability vector."""
    p = np.asarray(p, dtype=float)
    top = int(np.argmax(p))
    order = np.sort(p)[::-1]
    margin = float(order[0] - order[1]) if len(p) > 1 else 1.0
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    p_normal = float(p[classes.index("normal")]) if "normal" in classes else 0.0
    p_defect = 1.0 - p_normal

    defect_idx = [i for i, c in enumerate(classes) if c != "normal"]
    dp = p[defect_idx]
    if dp.sum() > 1e-9:
        cond = dp / dp.sum()
        defect_type = classes[defect_idx[int(np.argmax(dp))]]
        type_conf = float(cond.max())
        cond_entropy = float(-(cond * np.log(cond + 1e-12)).sum() / np.log(len(cond)))
    else:
        defect_type, type_conf, cond_entropy = "none", 0.0, 1.0

    uncertain = (p[top] < CONF_THRESHOLD) or (margin < MARGIN_THRESHOLD) or (entropy > ENTROPY_THRESHOLD)
    novel = (p_defect >= AUTO_REJECT_THRESHOLD and
             (type_conf < TYPE_CONF_THRESHOLD or cond_entropy > ENTROPY_THRESHOLD))

    if p_defect >= AUTO_REJECT_THRESHOLD:
        decision = "REJECT_NOVEL" if novel else "REJECT"
        reason = ("Defect probability is high but the defect type is ambiguous; not forcing a guess."
                  if novel else "Defect probability at or above the auto-reject threshold.")
    elif p_defect >= T_LOW:
        decision = "REVIEW"
        reason = (f"Borderline: P(defect)={p_defect:.2f} is above the cost-based accept limit "
                  f"{T_LOW:.2f} (false accepts cost {COST_FALSE_ACCEPT / COST_FALSE_REJECT:.0f}x more).")
    elif uncertain:
        decision = "REVIEW"
        reason = "Low confidence / high entropy even though P(defect) is small."
    else:
        decision = "PASS"
        reason = "Confidently normal."

    return {
        "top_class": classes[top], "top_conf": float(p[top]), "margin": margin,
        "entropy": entropy, "p_normal": p_normal, "p_defect": p_defect,
        "defect_type": defect_type, "type_conf": type_conf, "cond_entropy": cond_entropy,
        "uncertain": bool(uncertain), "novel": bool(novel),
        "decision": decision, "reason": reason,
        "probs": {c: float(v) for c, v in zip(classes, p)},
    }


# ============================================================
# 7. DEFECT LOCALIZATION (occlusion sensitivity heatmap + box)
# ============================================================

def occlusion_heatmap(vision, img, class_idx):
    """Slide a grey patch over the image; regions whose occlusion lowers the class
    probability the most are where the evidence for the defect sits."""
    H, W = img.shape[:2]
    ph, pw = max(8, int(H / OCC_GRID)), max(8, int(W / OCC_GRID))
    sh, sw = max(4, int(ph * OCC_STRIDE)), max(4, int(pw * OCC_STRIDE))
    ys = list(range(0, max(H - ph, 0) + 1, sh))
    xs = list(range(0, max(W - pw, 0) + 1, sw))
    if ys[-1] != max(H - ph, 0):
        ys.append(max(H - ph, 0))
    if xs[-1] != max(W - pw, 0):
        xs.append(max(W - pw, 0))

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)     # log-odds stay informative even when P is ~100%
        return np.log(p) - np.log(1 - p)

    fill = img.mean(axis=(0, 1)).astype(np.uint8)
    base = vision.raw_probs([img])[0][class_idx]
    batch, coords = [], []
    for y in ys:
        for x in xs:
            im = img.copy()
            im[y:y + ph, x:x + pw] = fill
            batch.append(im)
            coords.append((y, x))
    probs = vision.raw_probs(batch)[:, class_idx]
    drops = np.clip(logit(base) - logit(probs), 0, None)

    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for (y, x), dv in zip(coords, drops):
        heat[y:y + ph, x:x + pw] += dv
        count[y:y + ph, x:x + pw] += 1
    heat = heat / np.maximum(count, 1)
    return heat, float(heat.max())


def localize_defect(vision, img, class_name, save_path=None):
    result = {"supported": False, "box": None, "area_fraction": None,
              "note": "", "max_drop": 0.0, "overlay": None}
    if class_name not in vision.classes:
        result["note"] = "Class unknown to the model."
        return result
    heat, max_drop = occlusion_heatmap(vision, img, vision.classes.index(class_name))
    result["max_drop"] = max_drop
    if max_drop < HEAT_MIN_SUPPORT:
        result["note"] = ("No image region measurably drives the prediction "
                          "(evidence is spread out); localization not supported.")
        return result
    heat_n = heat / max_drop
    ys, xs = np.where(heat_n >= HEAT_BOX_THRESHOLD)
    if len(xs) == 0:
        result["note"] = "Heatmap has no region above the box threshold."
        return result
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    H, W = img.shape[:2]
    area = (x2 - x1 + 1) * (y2 - y1 + 1) / float(H * W)
    result.update({"supported": True, "box": (x1, y1, x2, y2), "area_fraction": area})
    result["note"] = ("Diffuse evidence (large region); treat the box as approximate."
                      if area > HEAT_DIFFUSE_AREA else "Localized region found.")
    if save_path:
        hm = cv2.applyColorMap((heat_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, hm, 0.45, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(overlay, class_name.upper(), (max(x1, 5), max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, np.hstack([img, overlay]))
        result["overlay"] = save_path
    return result


# ============================================================
# 8. ROOT-CAUSE ZONE ATTRIBUTION
# ============================================================

def attribute_zone(defect, idx, ctx):
    priors = DEFECT_PRIORS.get(defect)
    if priors is None:
        return {"status": "NOT_DETERMINED", "top": "Not determined", "second": "",
                "confidence": "Low", "gap": 0.0, "probs": {}, "stress": {},
                "evidence": "", "reason": "No defect-specific zone model for this class."}

    b, m1 = ctx["baselines"], ctx["m1"]
    row = m1.iloc[idx]
    stress, util_z, wait_z = {}, {}, {}
    for zone in ZONES:
        util_z[zone] = float(np.clip(calc_z(row[zone + " Util"], *b[zone + " Util"]), -2, 4))
        wait_z[zone] = float(np.clip(calc_z(np.log1p(row[zone + " Waiting Time"]),
                                            *b[zone + " Wait"]), -2, 4))
        stress[zone] = (abs(util_z[zone]) + abs(wait_z[zone])) / 2

    storage_detail = ""
    if ctx["storage_cols"]:
        r2 = ctx["m2"].iloc[idx % len(ctx["m2"])]
        vals = []
        for key, col in ctx["storage_cols"].items():
            zz = float(np.clip(calc_z(np.log1p(max(float(r2[col]), 0)), *b[key]), -2, 4))
            vals.append(max(zz, 0.0))       # low dwell time is not a corrosion / WIP risk
        stress["Storage"] = float(np.mean(vals))
        storage_detail = f"storage dwell/WIP deviation={stress['Storage']:+.2f} SD"

    top_stress_zone = max(stress, key=stress.get)
    if stress[top_stress_zone] < STRESS_THRESHOLD:
        return {"status": "NOT_CLEAR", "top": "No clear contributing zone", "second": "",
                "confidence": "Low", "gap": 0.0, "probs": {}, "stress": stress,
                "evidence": "; ".join(f"{z}={v:.2f}" for z, v in stress.items()),
                "reason": "No production zone deviates strongly from the historical baseline."}

    raw = {z: priors.get(z, 0.0) * np.exp(K * stress[z]) for z in stress}
    total = sum(raw.values())
    probs = {z: raw[z] / total for z in raw}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0], ranked[1]
    gap = top[1] - second[1]
    confidence = "Low" if gap < 0.10 else ("Medium" if gap < 0.20 else "High")

    if top[0] == "Storage":
        evidence = f"Storage: {storage_detail}, combined stress={stress['Storage']:.2f}"
    else:
        evidence = (f"{top[0]}: utilization deviation={util_z[top[0]]:+.2f} SD, "
                    f"waiting-time deviation={wait_z[top[0]]:+.2f} SD, "
                    f"combined stress={stress[top[0]]:.2f}")
    if top_stress_zone != top[0]:
        evidence += (f"; note: the strongest raw process deviation is in {top_stress_zone} "
                     f"({stress[top_stress_zone]:.2f}), but this defect type points mainly to {top[0]}")
    return {"status": "ATTRIBUTED", "top": top[0], "second": second[0],
            "confidence": confidence, "gap": gap, "probs": probs, "stress": stress,
            "util_z": util_z, "wait_z": wait_z, "evidence": evidence,
            "reason": "Defect-specific prior combined with production-zone stress."}


# ============================================================
# 9. ECONOMICS, WHAT-IF AND MARGIN MODEL
# ============================================================

def economics(started, completed, stored, defect_rate=DEFECT_RATE_ASSUMED,
              rework_share=AVG_REWORKABLE):
    started = np.asarray(started, dtype=float)
    completed = np.asarray(completed, dtype=float)
    stored = np.asarray(stored, dtype=float)
    defective = completed * defect_rate
    reworked = defective * rework_share
    scrapped = defective - reworked
    sellable = completed - scrapped
    revenue = sellable * ECON["price"]
    cost = (started * (ECON["material"] + ECON["processing"])
            + reworked * ECON["rework_cost"] + stored * ECON["holding_cost"])
    profit = revenue - cost
    margin = np.where(revenue > 0, profit / np.where(revenue > 0, revenue, 1), np.nan)
    return {"revenue": revenue, "cost": cost, "profit": profit, "margin": margin,
            "scrapped": scrapped, "reworked": reworked}


def what_if(prod, row_idx, defect_rate=DEFECT_RATE_ASSUMED, rework_share=AVG_REWORKABLE):
    d = prod["model2"]
    row = d.iloc[row_idx]
    started, completed, stored = float(row[prod["in_col"]]), float(row[prod["out_col"]]), float(row["WIP_Stored"])
    loss_now = float(row["Throughput_Loss_Percent"]) if not np.isnan(row["Throughput_Loss_Percent"]) else 0.0
    zone = row["Bottleneck_Zone"]
    util_now = float(row[zone + "_Util_Pct"])

    centers, values = prod["loss_curves"][zone]
    loss_new = loss_now
    if len(centers) >= 2:
        loss_new = min(loss_now, float(np.interp(util_now / (1 + WHATIF_CAPACITY_GAIN), centers, values)))
    completed_b = started * (1 - loss_new / 100.0)
    stored_b = stored * (loss_new / loss_now) if loss_now > 0 else stored

    scen = {
        "Current": (started, completed, stored, defect_rate, loss_now),
        f"A: defects -{int(WHATIF_DEFECT_REDUCTION * 100)}%": (started, completed, stored, defect_rate * (1 - WHATIF_DEFECT_REDUCTION), loss_now),
        f"B: {zone} capacity +{int(WHATIF_CAPACITY_GAIN * 100)}%": (started, completed_b, stored_b, defect_rate, loss_new),
        "A + B combined": (started, completed_b, stored_b, defect_rate * (1 - WHATIF_DEFECT_REDUCTION), loss_new),
    }
    rows, base_profit, base_margin = [], None, None
    for name, (s, c, st, dr, ls) in scen.items():
        e = economics(s, c, st, dr, rework_share)
        profit, margin = float(e["profit"]), float(e["margin"])
        if base_profit is None:
            base_profit, base_margin = profit, margin
        rows.append({"Scenario": name, "Throughput_Loss_Pct": ls, "Completed": c,
                     "Profit": profit, "Margin_Pct": margin * 100,
                     "Profit_Change": profit - base_profit,
                     "Margin_Change_Pts": (margin - base_margin) * 100})
    return pd.DataFrame(rows)


def fit_margin_model(prod):
    """Predict run margin from process conditions (excludes the output count)."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except Exception:
        return None
    d = prod["model2"]
    e = economics(d[prod["in_col"]], d[prod["out_col"]], d["WIP_Stored"])
    y = pd.Series(e["margin"], index=d.index)
    feats = [prod["in_col"]] + [z + "_Util_Pct" for z in ZONES] + \
            [prod["cols"][z + "_wait"] for z in ZONES] + ["WIP_Stored"]
    feats += [v for v in (prod["stor"]["s1t"], prod["stor"]["s2t"]) if v]
    X = d[feats].fillna(0)
    ok = y.notna()
    Xtr, Xte, ytr, yte = train_test_split(X[ok], y[ok], test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=0, n_jobs=-1)
    model.fit(Xtr, ytr)
    return {"model": model, "features": feats, "r2": float(r2_score(yte, model.predict(Xte)))}


# ============================================================
# 10. RECOMMENDATIONS (advisory)
# ============================================================

ZONE_ACTIONS = {
    "Drilling": "Check drill-bit wear / tool life, feed and speed, and coolant flow; add an in-process "
                "hole-diameter check and consider parallel drilling capacity.",
    "Milling": "Check cutter wear, fixture clamping and vibration; add a surface-finish check after milling "
               "and review cutting parameters.",
    "Assembly": "Check press force / torque limits, part handling and fixturing; balance the line or add "
                "assembly capacity and limit upstream release to cap work-in-process.",
    "Storage": "Reduce dwell time between stages (FIFO, smaller buffers), and protect parts with "
               "humidity control or anti-corrosion packaging.",
}


def build_recommendations(rec, prod, whatif_df, drift_flags):
    recs = []
    zr = rec.get("zone_result")
    if zr and zr["status"] == "ATTRIBUTED":
        recs.append((1, f"[{zr['top']}] " + ZONE_ACTIONS[zr["top"]],
                     f"{zr['evidence']} (zone confidence: {zr['confidence']})"))
        if zr["confidence"] == "Low":
            recs.append((2, f"Also inspect {zr['second']}: zone evidence is ambiguous between the top two.",
                         f"probability gap {zr['gap']:.2f}"))
    elif zr and zr["status"] == "NOT_CLEAR":
        recs.append((2, "No production zone stands out; inspect incoming material and handling before "
                        "adjusting any process.", zr["evidence"]))
    if rec.get("decision") in ("REVIEW", "REJECT_NOVEL"):
        recs.append((1, "Send the unit to a human inspector and add it to the labelling queue for model retraining.",
                     rec.get("decision_reason", "")))
    b = prod["primary"]
    recs.append((2, f"Production flow is constrained at {b} ({prod['high_util'][b]} runs at >=90% utilization). "
                    f"{ZONE_ACTIONS[b].split(';')[-1].strip().capitalize()}",
                 f"average throughput loss {prod['avg_loss_pct']:.2f}%"))
    if whatif_df is not None and len(whatif_df) > 1:
        best = whatif_df.iloc[1:].sort_values("Profit_Change", ascending=False).iloc[0]
        recs.append((3, f"Simulated best lever for this run: {best['Scenario']} "
                        f"(profit {best['Profit_Change']:+,.0f}, margin {best['Margin_Change_Pts']:+.1f} pts).",
                     "what-if using assumed economics"))
    for f in drift_flags:
        recs.append((2, "Investigate batch-to-batch process drift before the next run.", f))
    recs.sort(key=lambda r: r[0])
    return recs


# ============================================================
# 11. INSPECT ONE PRODUCT
# ============================================================

def inspect_product(idx, ctx, image_path=None, heatmap=True, verbose=True):
    vision, m1, prod = ctx["vision"], ctx["m1"], ctx["prod"]
    if image_path is None:
        image_path = ctx["image_paths"][idx % len(ctx["image_paths"])]
    true_label = os.path.basename(os.path.dirname(image_path)).lower()
    batch_id = m1.iloc[idx]["Batch_ID"]
    img = read_image(image_path)

    raw = vision.raw_probs([img])[0]
    p = vision.calibrated(raw)
    a = analyse_probs(p, vision.classes)

    d2 = prod["model2"]
    j = idx % len(d2)
    row2 = d2.iloc[j]
    rec = {
        "Product_Index": idx, "Batch_ID": batch_id, "Image": image_path,
        "Folder_Label": true_label, "Decision": a["decision"],
        "Top_Class": a["top_class"], "Defect": a["defect_type"] if a["p_defect"] >= T_LOW else "normal",
        "YOLO_Confidence": a["top_conf"], "P_Defect": a["p_defect"],
        "Margin_Top1_Top2": a["margin"], "Entropy": a["entropy"], "Type_Confidence": a["type_conf"],
        "Novel_Flag": a["novel"], "decision_reason": a["reason"], "Temperature": vision.T,
    }

    # ---- vision report ----
    if verbose:
        banner("PRODUCT INSPECTION")
        print(f"Product index : {idx}   Batch ID: {batch_id}")
        print(f"Image         : {image_path}   (folder label: {true_label})")
        print("\nVISION INSPECTION")
        print(f"Top class          : {a['top_class'].upper()}  (calibrated confidence {a['top_conf']:.2%}, T={vision.T:.1f})")
        print(f"P(defect)          : {a['p_defect']:.2%}   margin top1-top2: {a['margin']:.2f}   entropy: {a['entropy']:.2f}")
        print("Class probabilities: " + ", ".join(f"{c}={v:.2f}" for c, v in a["probs"].items()))
        print(f"DECISION           : {DECISION_TEXT[a['decision']]}")
        print(f"Why                : {a['reason']}")

    zone_result, loc, whatif_df = None, None, None
    is_defective_call = a["decision"] in ("REJECT", "REJECT_NOVEL", "REVIEW") and a["p_defect"] >= T_LOW

    # ---- localization ----
    if is_defective_call and heatmap:
        target = a["defect_type"]
        loc = localize_defect(vision, img, target,
                              os.path.join(LOCALIZATION_DIR, f"product_{idx}_{target}.png"))
        rec.update({"Localization_Supported": loc["supported"],
                    "Localization_Box": str(loc["box"]) if loc["box"] else "",
                    "Localization_Area": loc["area_fraction"], "Localization_Image": loc["overlay"] or ""})
        if verbose:
            print("\nDEFECT LOCALIZATION (weakly supervised heatmap)")
            print(f"Supported : {loc['supported']}   {loc['note']}")
            if loc["box"]:
                print(f"Box (x1,y1,x2,y2): {loc['box']}   area = {loc['area_fraction']:.1%} of image")
                print("Overlay saved to :", loc["overlay"])

    # ---- root cause ----
    if is_defective_call and a["decision"] != "REJECT_NOVEL":
        zone_result = attribute_zone(a["defect_type"], idx, ctx)
    elif a["decision"] == "REJECT_NOVEL":
        zone_result = {"status": "NOT_DETERMINED", "top": "Not determined", "second": "",
                       "confidence": "Low", "gap": 0.0, "probs": {}, "stress": {}, "evidence": "",
                       "reason": "Defect type is uncertain / possibly novel; zone attribution withheld."}
    if zone_result:
        rec.update({
            "Likely_Zone": zone_result["top"], "Second_Zone": zone_result["second"],
            "Zone_Confidence": zone_result["confidence"], "Probability_Gap": zone_result["gap"],
            "Zone_Status": zone_result["status"], "Reason": zone_result["reason"],
            "Evidence": zone_result["evidence"]})
        for z in ZONES + ["Storage"]:
            rec[z + "_Stress"] = zone_result["stress"].get(z, np.nan)
            rec[z + "_Contribution"] = zone_result["probs"].get(z, np.nan)
        if verbose:
            print("\nROOT-CAUSE ASSISTANT")
            print(f"LIKELY CONTRIBUTING ZONE : {zone_result['top'].upper()}")
            print(f"ZONE CONFIDENCE          : {zone_result['confidence']}")
            print(f"Why: {zone_result['reason']}")
            if zone_result["evidence"]:
                print("Evidence:", zone_result["evidence"])
            if zone_result["probs"]:
                print("Zone contribution: " + ", ".join(
                    f"{z} {v * 100:.1f}%" for z, v in sorted(zone_result["probs"].items(), key=lambda kv: -kv[1])))
            print("NOTE: rule/score based (defect prior + production abnormality); it points to the most "
                  "likely zone, it does not prove the cause.")
    else:
        rec.update({"Likely_Zone": "N/A", "Zone_Confidence": "N/A", "Zone_Status": "N/A",
                    "Reason": a["reason"], "Evidence": ""})

    # ---- flow, drift and economics for this run ----
    drift_flags = drift_check(m1, idx)
    rec["Drift_Flags"] = " | ".join(drift_flags)
    rework = REWORKABLE_SHARE.get(a["defect_type"], DEFAULT_REWORKABLE)
    whatif_df = what_if(prod, j, rework_share=rework if is_defective_call else AVG_REWORKABLE)
    base = whatif_df.iloc[0]
    unit_loss = ((1 - rework) * (ECON["material"] + ECON["processing"]) + rework * ECON["rework_cost"]
                 if is_defective_call else 0.0)
    rec.update({
        "Run_Bottleneck_Zone": row2["Bottleneck_Zone"],
        "Run_Throughput_Loss_Pct": row2["Throughput_Loss_Percent"],
        "Run_WIP_Stored": row2["WIP_Stored"],
        "Run_Profit": base["Profit"], "Run_Margin_Pct": base["Margin_Pct"],
        "Unit_Defect_Cost": unit_loss,
    })
    for _, r in whatif_df.iloc[1:].iterrows():
        rec["WhatIf_" + r["Scenario"].split(":")[0].strip().replace(" ", "")] = r["Profit_Change"]
    mm = ctx.get("margin_model")
    if mm:
        Xrow = d2.iloc[[j]][mm["features"]].fillna(0)
        rec["Predicted_Margin_Pct"] = float(mm["model"].predict(Xrow)[0]) * 100

    if verbose:
        print("\nPRODUCTION FLOW FOR THIS RUN")
        print(f"Bottleneck zone (this run): {row2['Bottleneck_Zone']}   "
              f"throughput loss: {row2['Throughput_Loss_Percent']:.2f}%   parts in storage: {row2['WIP_Stored']:.0f}")
        print(f"Plant-wide primary bottleneck: {prod['primary']}   secondary: {prod['secondary']}")
        print("Batch drift: " + (" ; ".join(drift_flags) if drift_flags else "none detected"))
        print("\nECONOMICS (ASSUMED parameters - replace with organizer data)")
        print(f"Run profit: {base['Profit']:,.0f}   margin: {base['Margin_Pct']:.1f}%"
              + (f"   predicted margin (model): {rec['Predicted_Margin_Pct']:.1f}%" if mm else ""))
        if is_defective_call:
            print(f"Expected cost of this defective unit: {unit_loss:.1f} (rework share {rework:.0%})")
        print(whatif_df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
        print("(What-if losses are read from the observed utilization -> throughput-loss curve; treat as an optimistic estimate.)")

    recs = build_recommendations({"zone_result": zone_result, "decision": a["decision"],
                                  "decision_reason": a["reason"]}, prod, whatif_df, drift_flags)
    rec["Recommendations"] = " | ".join(f"P{p_}: {t}" for p_, t, _ in recs)
    if verbose:
        print("\nRECOMMENDATIONS (advisory / simulated)")
        for pr, text, ev in recs:
            print(f" P{pr} {text}")
            if ev:
                print(f"     evidence: {ev}")
    return rec


# ============================================================
# 12. EVALUATION: accuracy, calibration, FA/FR, robustness, OOD
# ============================================================

def ece_score(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y)
    edges = np.linspace(0, 1, bins + 1)
    ece, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rel.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(ece), rel


def fit_temperature(p_raw, y):
    best_nll, best_t = 1e9, 1.0
    for T in TEMPERATURE_GRID:
        q = apply_temperature(p_raw, T)
        nll = -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll - 1e-9:
            best_nll, best_t = nll, float(T)
    return round(best_t, 2)


def decision_rates(p_cal, y, classes):
    """False accept / false reject / review rates using the full three-way decision."""
    normal_idx = classes.index("normal") if "normal" in classes else -1
    decisions = [analyse_probs(row, classes)["decision"] for row in p_cal]
    is_def = (y != normal_idx)
    dec = np.array(decisions)
    n_def, n_norm = max(int(is_def.sum()), 1), max(int((~is_def).sum()), 1)
    far = float(((dec == "PASS") & is_def).sum() / n_def)
    frr = float((np.isin(dec, ["REJECT", "REJECT_NOVEL"]) & ~is_def).sum() / n_norm)
    review = float((dec == "REVIEW").mean())
    auto = dec != "REVIEW"
    auto_correct = float((((dec == "PASS") & ~is_def) | (np.isin(dec, ["REJECT", "REJECT_NOVEL"]) & is_def))[auto].mean()) if auto.any() else float("nan")
    return {"false_accept_rate": far, "false_reject_rate": frr, "review_rate": review,
            "auto_decision_accuracy": auto_correct, "decisions": decisions}


def _bright(f):
    return lambda im: np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)


def _contrast(f):
    return lambda im: np.clip((im.astype(np.float32) - 127.5) * f + 127.5, 0, 255).astype(np.uint8)


def _rotate(angle):
    def fn(im):
        h, w = im.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return fn


def _noise(sigma):
    rng = np.random.default_rng(0)
    return lambda im: np.clip(im.astype(np.float32) + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def _jpeg(q):
    def fn(im):
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return fn


PERTURBATIONS = {
    "dim light (x0.6)": _bright(0.6),
    "bright light (x1.5)": _bright(1.5),
    "low contrast": _contrast(0.6),
    "rotate 15 deg": _rotate(15),
    "rotate 90 deg": lambda im: np.rot90(im).copy(),
    "horizontal flip": lambda im: im[:, ::-1].copy(),
    "blur": lambda im: cv2.GaussianBlur(im, (9, 9), 0),
    "sensor noise": _noise(15),
    "JPEG compression": _jpeg(20),
}


def ood_probes(shape):
    """Synthetic images that are not any known part: a good system should not say 'confidently normal'."""
    rng = np.random.default_rng(1)
    h, w = shape[:2]
    probes = {
        "random noise": rng.integers(0, 256, shape, dtype=np.uint8),
        "flat grey": np.full(shape, 127, dtype=np.uint8),
        "black": np.zeros(shape, dtype=np.uint8),
        "white": np.full(shape, 255, dtype=np.uint8),
    }
    checker = ((np.indices((h, w)).sum(axis=0) // 16) % 2 * 255).astype(np.uint8)
    probes["checkerboard"] = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    blocks = np.zeros(shape, dtype=np.uint8)
    for _ in range(12):
        y, x = rng.integers(0, h - 10), rng.integers(0, w - 10)
        blocks[y:y + rng.integers(10, h // 3), x:x + rng.integers(10, w // 3)] = rng.integers(0, 256, 3)
    probes["random colour blocks"] = blocks
    return probes


def evaluate(vision, verbose=True):
    items = [(p_, l) for p_, l in list_labeled_images(TEST_DIR) if l in vision.classes]
    if not items:
        raise RuntimeError("No labelled test images found. Check TEST_DIR and folder names.")
    if len(items) > MAX_EVAL_IMAGES:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(items), MAX_EVAL_IMAGES, replace=False))
        items = [items[i] for i in keep]
    imgs = [read_image(p_) for p_, _ in items]
    y = np.array([vision.classes.index(l) for _, l in items])
    raw = vision.raw_probs(imgs)

    # ---- calibration set (validation folder if present, otherwise half of the test set) ----
    val_items = [(p_, l) for p_, l in list_labeled_images(VAL_DIR) if l in vision.classes] if os.path.isdir(VAL_DIR) else []
    if val_items:
        raw_cal = vision.raw_probs([read_image(p_) for p_, _ in val_items])
        y_cal = np.array([vision.classes.index(l) for _, l in val_items])
        eval_mask = np.ones(len(y), dtype=bool)
        cal_source = f"validation folder ({len(val_items)} images)"
    else:
        cal_mask = np.arange(len(y)) % 2 == 0
        raw_cal, y_cal = raw[cal_mask], y[cal_mask]
        eval_mask = ~cal_mask
        cal_source = "half of the test set (no validation folder found); evaluated on the other half"
    T = fit_temperature(raw_cal, y_cal)
    vision.T = T
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"temperature": T}, f)

    imgs_e = [im for im, m in zip(imgs, eval_mask) if m]
    raw_e, y_e = raw[eval_mask], y[eval_mask]
    p_cal = apply_temperature(raw_e, T)

    # ---- classification quality ----
    n_cls = len(vision.classes)
    pred = p_cal.argmax(1)
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, q in zip(y_e, pred):
        conf[t, q] += 1
    per_class = []
    for i, c in enumerate(vision.classes):
        tp = conf[i, i]
        prec = tp / max(conf[:, i].sum(), 1)
        rec_ = tp / max(conf[i, :].sum(), 1)
        f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
        per_class.append({"class": c, "precision": prec, "recall": rec_, "f1": f1, "support": int(conf[i, :].sum())})
    accuracy = float((pred == y_e).mean())
    ece_raw, _ = ece_score(raw_e, y_e)
    ece_cal, reliability = ece_score(p_cal, y_e)
    dr = decision_rates(p_cal, y_e, vision.classes)

    # ---- threshold sweep (false accept vs false reject trade-off) ----
    normal_idx = vision.classes.index("normal") if "normal" in vision.classes else -1
    is_def = y_e != normal_idx
    p_def = 1 - p_cal[:, normal_idx] if normal_idx >= 0 else 1 - p_cal.max(1)
    sweep = []
    for t in [0.05, 0.10, round(T_LOW, 3), 0.25, 0.35, 0.50, 0.65, 0.80]:
        rej = p_def >= t
        far = float((~rej & is_def).sum() / max(is_def.sum(), 1))
        frr = float((rej & ~is_def).sum() / max((~is_def).sum(), 1))
        cost = float(((~rej & is_def).sum() * COST_FALSE_ACCEPT + (rej & ~is_def).sum() * COST_FALSE_REJECT) / len(y_e))
        sweep.append({"reject_threshold": t, "false_accept_rate": far, "false_reject_rate": frr, "expected_cost": cost})

    # ---- robustness to changed inspection conditions ----
    robust = []
    for name, fn in PERTURBATIONS.items():
        pert = [fn(im) for im in imgs_e]
        pp = apply_temperature(vision.raw_probs(pert), T)
        rr = decision_rates(pp, y_e, vision.classes)
        robust.append({"condition": name, "accuracy": float((pp.argmax(1) == y_e).mean()),
                       "false_accept_rate": rr["false_accept_rate"], "false_reject_rate": rr["false_reject_rate"],
                       "review_rate": rr["review_rate"], "mean_confidence": float(pp.max(1).mean())})

    # ---- out-of-distribution probes ----
    probes = ood_probes(imgs_e[0].shape)
    pp = apply_temperature(vision.raw_probs(list(probes.values())), T)
    ood = []
    for (name, _), row in zip(probes.items(), pp):
        a = analyse_probs(row, vision.classes)
        ood.append({"probe": name, "decision": a["decision"], "top_class": a["top_class"],
                    "top_conf": a["top_conf"], "flagged": a["decision"] != "PASS"})
    ood_flag_rate = float(np.mean([o["flagged"] for o in ood]))

    ev = {
        "n_eval": int(len(y_e)), "calibration_source": cal_source, "temperature": T,
        "accuracy": accuracy, "ece_before": ece_raw, "ece_after": ece_cal,
        "per_class": per_class, "confusion": conf.tolist(), "classes": vision.classes,
        "reliability": reliability, "decision_rates": {k: v for k, v in dr.items() if k != "decisions"},
        "threshold_sweep": sweep, "robustness": robust, "ood_probes": ood, "ood_flag_rate": ood_flag_rate,
    }
    with open(EVAL_FILE, "w") as f:
        json.dump(jsonable(ev), f, indent=2)

    if verbose:
        banner("MODEL EVALUATION")
        print(f"Evaluated on {ev['n_eval']} images.  Calibration set: {cal_source}")
        print(f"Top-1 accuracy: {accuracy:.2%}    ECE before/after calibration: {ece_raw:.3f} / {ece_cal:.3f}   (T={T})")
        print("\nPer class:")
        print(pd.DataFrame(per_class).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nConfusion matrix (rows = actual, columns = predicted):")
        print(pd.DataFrame(conf, index=vision.classes, columns=vision.classes).to_string())
        print(f"\nDecision system (accept / review / reject):")
        print(f"  False-accept rate (defect passed): {dr['false_accept_rate']:.2%}")
        print(f"  False-reject rate (good rejected) : {dr['false_reject_rate']:.2%}")
        print(f"  Sent to manual review             : {dr['review_rate']:.2%}")
        print("\nThreshold trade-off (single cut-off on P(defect)):")
        print(pd.DataFrame(sweep).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nRobustness (changed inspection conditions):")
        print(pd.DataFrame(robust).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nOut-of-distribution probes flagged (not PASS): {ood_flag_rate:.0%}")
        print(pd.DataFrame(ood).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\nReport saved to:", EVAL_FILE)
    return ev


# ============================================================
# 13. DASHBOARD (self-contained HTML)
# ============================================================

def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, alt=""):
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


def file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:                      # a chart problem must never break the dashboard
        return f"<p class='muted'>Chart unavailable ({type(exc).__name__}: {exc})</p>"


def chart_zone_health(prod):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    zs = prod["zone_summary"]
    ax[0].bar(zs["Zone"], zs["Average_Utilization_Percent"], color="#3b6ea5")
    ax[0].set_title("Average utilization (%)")
    for i, v in enumerate(zs["Average_Utilization_Percent"]):
        ax[0].text(i, v + 1, f"{v:.0f}", ha="center")
    rk = prod["ranking"]
    ax[1].bar(rk["Zone"], rk["Bottleneck_Percentage"], color="#c0504d")
    ax[1].set_title("Share of runs where zone is the bottleneck (%)")
    return img_tag(b64fig(fig), "zone health")


def chart_util_wait(prod):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    uw = prod["util_wait"]
    for zone in ZONES:
        g = uw[uw["Zone"] == zone]
        ax.plot(g["Utilization_Range"], g["Average_Waiting_Time"], marker="o", label=zone)
    ax.set_title("Average waiting time by utilization range")
    ax.set_xlabel("Utilization range")
    ax.set_ylabel("Waiting / queue time")
    ax.legend()
    ax.set_yscale("symlog")
    return img_tag(b64fig(fig), "utilization vs waiting")


def chart_outcomes(bdf):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    dc = bdf["Decision"].value_counts()
    ax[0].bar([DECISION_TEXT[k].split(" (")[0] for k in dc.index], dc.values, color="#3b6ea5")
    ax[0].set_title("Inspection outcomes")
    ax[0].tick_params(axis="x", rotation=20)
    defects = bdf[bdf["Decision"].isin(["REJECT", "REJECT_NOVEL", "REVIEW"])]["Top_Class"].value_counts()
    ax[1].bar(defects.index, defects.values, color="#c0504d")
    ax[1].set_title("Flagged units by predicted class")
    return img_tag(b64fig(fig), "outcomes")


def chart_rolling(bdf):
    flag = (bdf["P_Defect"] >= 0.5).astype(float)
    roll = flag.rolling(10, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(range(len(roll)), roll * 100, color="#c0504d")
    ax.set_title("Rolling defect rate (last 10 units)")
    ax.set_xlabel("Units inspected")
    ax.set_ylabel("% predicted defective")
    return img_tag(b64fig(fig), "rolling defect rate")


def chart_zone_mix(bdf):
    d = bdf[bdf["Zone_Status"] == "ATTRIBUTED"]
    if d.empty:
        return "<p class='muted'>No attributed defects in this batch.</p>"
    ct = pd.crosstab(d["Defect"], d["Likely_Zone"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Likely contributing zone by defect type")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    return img_tag(b64fig(fig), "zone mix")


def chart_margin(prod):
    d = prod["model2"]
    e = economics(d[prod["in_col"]], d[prod["out_col"]], d["WIP_Stored"])
    ideal = economics(d[prod["in_col"]], d[prod["in_col"]], 0)
    m = pd.Series(e["margin"]).dropna() * 100
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(m, bins=30, color="#3b6ea5")
    ax.axvline(m.mean(), color="#c0504d", label=f"mean {m.mean():.1f}%")
    ax.axvline(float(np.nanmean(ideal["margin"])) * 100, color="green", linestyle="--", label="no-constraint ideal")
    ax.set_title("Estimated margin across production runs (assumed economics)")
    ax.set_xlabel("Margin (%)")
    ax.legend()
    return img_tag(b64fig(fig), "margin")


def chart_whatif(bdf):
    cols = [c for c in bdf.columns if c.startswith("WhatIf_")]
    if not cols:
        return ""
    means = bdf[cols].mean()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar([c.replace("WhatIf_", "") for c in cols], means.values, color="#4f9d5d")
    ax.set_title("Average simulated profit change per run")
    return img_tag(b64fig(fig), "what-if")


def chart_confusion(ev):
    conf = np.array(ev["confusion"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(ev["classes"])))
    ax.set_yticks(range(len(ev["classes"])))
    ax.set_xticklabels(ev["classes"], rotation=45)
    ax.set_yticklabels(ev["classes"])
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, conf[i, j], ha="center", va="center")
    ax.set_title("Confusion matrix")
    return img_tag(b64fig(fig), "confusion matrix")


def chart_reliability(ev):
    rel = ev["reliability"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    if rel:
        ax.plot([r[0] for r in rel], [r[1] for r in rel], marker="o", label="model")
    ax.set_title(f"Reliability (ECE {ev['ece_after']:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return img_tag(b64fig(fig), "reliability")


def chart_robust(ev):
    r = pd.DataFrame(ev["robustness"])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(r["condition"], r["accuracy"] * 100, color="#3b6ea5")
    ax.axhline(ev["accuracy"] * 100, color="green", linestyle="--", label="clean images")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy under changed inspection conditions")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    return img_tag(b64fig(fig), "robustness")


def html_table(df, floatfmt="{:,.2f}"):
    return df.to_html(index=False, classes="tbl", border=0, float_format=lambda v: floatfmt.format(v), na_rep="")


def build_dashboard(ctx, bdf, ev=None):
    prod = ctx["prod"]
    n = len(bdf)
    counts = bdf["Decision"].value_counts().to_dict()
    flagged = int((bdf["P_Defect"] >= 0.5).sum())
    avg_margin = float(bdf["Run_Margin_Pct"].mean())
    cards = [
        ("Units inspected", n), ("Pass", counts.get("PASS", 0)),
        ("Reject", counts.get("REJECT", 0) + counts.get("REJECT_NOVEL", 0)),
        ("Manual review", counts.get("REVIEW", 0)),
        ("Predicted defect rate", f"{flagged / max(n, 1):.0%}"),
        ("Primary bottleneck", prod["primary"]),
        ("Avg throughput loss", f"{prod['avg_loss_pct']:.1f}%"),
        ("Avg run margin (assumed)", f"{avg_margin:.1f}%"),
    ]
    if ev:
        dr = ev["decision_rates"]
        cards += [("Test accuracy", f"{ev['accuracy']:.1%}"),
                  ("False-accept rate", f"{dr['false_accept_rate']:.1%}"),
                  ("False-reject rate", f"{dr['false_reject_rate']:.1%}"),
                  ("Calibration (ECE)", f"{ev['ece_after']:.3f}")]
    card_html = "".join(f"<div class='card'><div class='v'>{v}</div><div class='k'>{k}</div></div>" for k, v in cards)

    latest = bdf[bdf["Zone_Status"].isin(["ATTRIBUTED", "NOT_CLEAR", "NOT_DETERMINED"])].tail(12)
    cols = [c for c in ["Product_Index", "Top_Class", "YOLO_Confidence", "Decision", "Likely_Zone",
                        "Zone_Confidence", "Evidence"] if c in latest.columns]
    root_tbl = html_table(latest[cols]) if len(latest) else "<p class='muted'>No defective units in this batch.</p>"

    gallery = ""
    if "Localization_Image" in bdf.columns:
        paths = [p_ for p_ in bdf["Localization_Image"].dropna().tolist() if p_ and os.path.exists(p_)][:3]
        gallery = "".join(f"<div>{img_tag(file_b64(p_), 'localization')}<p class='muted'>{os.path.basename(p_)}</p></div>" for p_ in paths)

    top_recs = []
    if "Recommendations" in bdf.columns:
        seen = {}
        for s in bdf["Recommendations"].dropna():
            for item in str(s).split(" | "):
                seen[item] = seen.get(item, 0) + 1
        top_recs = sorted(seen.items(), key=lambda kv: -kv[1])[:8]
    rec_html = "".join(f"<li>{t} <span class='muted'>(x{c})</span></li>" for t, c in top_recs)

    mm = ctx.get("margin_model")
    margin_note = (f"Margin prediction model (random forest on process conditions): R2 = {mm['r2']:.2f} on held-out runs."
                   if mm else "Margin prediction model unavailable (install scikit-learn).")

    ev_html = ""
    if ev:
        ev_html = f"""
        <h2>Model reliability and robustness</h2>
        <div class='row'>{_safe(chart_confusion, ev)}{_safe(chart_reliability, ev)}</div>
        {_safe(chart_robust, ev)}
        <h3>Per-class results</h3>{html_table(pd.DataFrame(ev['per_class']), '{:.3f}')}
        <h3>False-accept / false-reject trade-off</h3>{html_table(pd.DataFrame(ev['threshold_sweep']), '{:.3f}')}
        <h3>Out-of-distribution probes</h3>
        <p class='muted'>{ev['ood_flag_rate']:.0%} of synthetic non-part images were flagged instead of confidently passed.</p>
        {html_table(pd.DataFrame(ev['ood_probes']), '{:.2f}')}
        <p class='muted'>Calibration: {ev['calibration_source']}. Temperature = {ev['temperature']}.</p>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Inspection Dashboard</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#1f2933}}
header{{background:#1f3a5f;color:#fff;padding:18px 32px}} header p{{margin:4px 0 0;opacity:.8}}
main{{max-width:1050px;margin:auto;padding:16px 24px}} h2{{border-bottom:2px solid #1f3a5f;padding-bottom:4px;margin-top:34px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}} .card{{background:#fff;border-radius:8px;padding:12px 16px;min-width:130px;box-shadow:0 1px 3px #0002}}
.card .v{{font-size:22px;font-weight:600;color:#1f3a5f}} .card .k{{font-size:12px;color:#66788a}}
.row{{display:flex;gap:16px;flex-wrap:wrap}} .row>div{{flex:1 1 320px;min-width:0}} img{{max-width:100%;background:#fff;border-radius:6px}}
.tbl{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}} .tbl th{{background:#1f3a5f;color:#fff;padding:6px;text-align:left}}
.tbl td{{padding:5px 6px;border-bottom:1px solid #e2e8f0}} .muted{{color:#66788a;font-size:12px}}
.note{{background:#fff8e1;border-left:4px solid #f0b400;padding:8px 12px;margin:12px 0}}
</style></head><body>
<header><h1 style="margin:0">Visual Inspection &amp; Defect Root-Cause Assistant</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; all recommendations and profit figures are simulated / advisory</p></header>
<main>
<div class="cards">{card_html}</div>

<h2>1. Product quality (continuously updated)</h2>
{_safe(chart_outcomes, bdf)}{_safe(chart_rolling, bdf)}

<h2>2. Production-flow health</h2>
{_safe(chart_zone_health, prod)}{_safe(chart_util_wait, prod)}
<h3>Zone performance</h3>{html_table(prod['zone_summary'])}
<p class="muted">Average input {prod['avg_in']:.0f} parts, output {prod['avg_out']:.0f} parts, throughput loss
{prod['avg_loss_pct']:.2f}%, average parts held in storage {prod['avg_wip']:.0f}.</p>

<h2>3. Defect root cause</h2>
{_safe(chart_zone_mix, bdf)}
<h3>Latest attributions</h3>{root_tbl}
{('<h3>Defect localization (original | heatmap + box)</h3><div class="row">' + gallery + '</div>') if gallery else ''}

<h2>4. Profitability (assumed economics)</h2>
{_safe(chart_margin, prod)}{_safe(chart_whatif, bdf)}
<p class="muted">{margin_note}</p>

<h2>5. Recommended actions (advisory)</h2><ul>{rec_html}</ul>
{ev_html}

<h2>Assumptions and limitations</h2>
<div class="note"><ul>
<li>Images and production runs are paired by index for the demo; no dataset links a specific image to a specific run.</li>
<li>Zone attribution is rule/score based (defect prior x production stress). It ranks likely zones; it does not prove a cause.</li>
<li>Localization is a weakly supervised heatmap (no ground-truth boxes were provided), so its precision cannot be measured.</li>
<li>Economic parameters (price, cost, rework, holding cost) and the base defect rate are assumptions, to be replaced by organizer data.</li>
<li>Calibration uses temperature scaling with T&gt;=1 (never sharpens confidence). The test set is class-balanced, so its accuracy is not a production defect-rate estimate.</li>
<li>Robustness tests are synthetic perturbations; they approximate, not replace, real changes in lighting and part orientation.</li>
</ul></div>
</main></body></html>"""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print("\nDashboard saved to:", DASHBOARD_FILE)


# ============================================================
# 14. MODES
# ============================================================

def build_context():
    banner("LOADING")
    m1, m2 = load_production_data()
    if m2 is None:
        raise FileNotFoundError("Model_2.csv is required for bottleneck, storage and economics analysis.")
    prod = analyse_production(m2)
    baselines, storage_cols = build_baselines(m1, m2, prod)
    margin_model = fit_margin_model(prod)

    items = list_labeled_images(TEST_DIR)
    if not items:
        raise RuntimeError("No test images found. Check TEST_DIR.")
    image_paths = [p_ for p_, _ in items]
    print("\nTotal test images:", len(image_paths))

    T = 1.0
    if os.path.exists(CALIBRATION_FILE):
        try:
            T = float(json.load(open(CALIBRATION_FILE))["temperature"])
            print("Loaded calibration temperature:", T)
        except Exception:
            pass
    print("Loading YOLO model...")
    vision = Vision(MODEL_PATH, temperature=T)
    print("Classes:", vision.classes)
    return {"m1": m1, "m2": m2, "prod": prod, "baselines": baselines, "storage_cols": storage_cols,
            "margin_model": margin_model, "image_paths": image_paths, "vision": vision}


def run_single(ctx, idx=None):
    n = len(ctx["m1"])
    if idx is None:
        print(f"\nValid product numbers: 0 to {n - 1}")
        text = input("Enter production row/product number (e.g. 890 or BATCH_0890): ").strip()
        text = text.upper().replace("BATCH_", "")
        idx = int(text)
    if idx < 0 or idx >= n:
        raise IndexError(f"Product number must be between 0 and {n - 1}.")
    rec = inspect_product(idx, ctx, heatmap=True, verbose=True)
    pd.DataFrame([rec]).to_csv(OUTPUT_FILE, index=False)
    print("\nRESULT SAVED:", OUTPUT_FILE)


def run_batch(ctx, n_products=60, ev=None, heatmaps=5):
    banner(f"BATCH INSPECTION ({n_products} products)")
    m1_len = len(ctx["m1"])
    idxs = np.linspace(0, m1_len - 1, n_products, dtype=int)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(ctx["image_paths"]))
    records, used_heat = [], 0
    for k, idx in enumerate(idxs):
        img_path = ctx["image_paths"][order[k % len(order)]]
        want_heat = used_heat < heatmaps
        rec = inspect_product(int(idx), ctx, image_path=img_path, heatmap=want_heat, verbose=False)
        if rec.get("Localization_Image"):
            used_heat += 1
        records.append(rec)
        flagged = sum(r["P_Defect"] >= 0.5 for r in records)
        print(f"[{k + 1:3d}/{n_products}] product {idx:4d}  {rec['Top_Class']:8s} "
              f"{DECISION_TEXT[rec['Decision']]:38s} zone: {str(rec['Likely_Zone']):12s} "
              f"| running defect rate {flagged / len(records):.0%}")
    bdf = pd.DataFrame(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    bdf.to_csv(BATCH_OUTPUT_FILE, index=False)
    print("\nBatch results saved to:", BATCH_OUTPUT_FILE)
    banner("BATCH SUMMARY")
    print(bdf["Decision"].map(DECISION_TEXT).value_counts().to_string())
    attributed = bdf[bdf["Zone_Status"] == "ATTRIBUTED"]
    if len(attributed):
        print("\nLikely contributing zone (attributed defects):")
        print(attributed["Likely_Zone"].value_counts().to_string())
    print(f"\nAverage run margin (assumed economics): {bdf['Run_Margin_Pct'].mean():.1f}%")
    build_dashboard(ctx, bdf, ev)
    return bdf


def run_eval(ctx):
    return evaluate(ctx["vision"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="menu", choices=["menu", "single", "batch", "eval", "demo"])
    parser.add_argument("--product", type=int, default=None)
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()

    ctx = build_context()
    mode = args.mode
    if mode == "menu":
        print("\n1) Inspect a single product\n2) Batch inspection + dashboard\n"
              "3) Model evaluation (accuracy, calibration, robustness)\n4) Full demo (evaluation + batch + dashboard)")
        mode = {"1": "single", "2": "batch", "3": "eval", "4": "demo"}.get(input("Choose 1-4: ").strip(), "single")

    if mode == "single":
        run_single(ctx, args.product)
    elif mode == "eval":
        run_eval(ctx)
    elif mode == "batch":
        ev = None
        if os.path.exists(EVAL_FILE):
            ev = json.load(open(EVAL_FILE))
        run_batch(ctx, args.n, ev)
    elif mode == "demo":
        ev = run_eval(ctx)
        run_batch(ctx, args.n, ev)


if __name__ == "__main__":
    main()