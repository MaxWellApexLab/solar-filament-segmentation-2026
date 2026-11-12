"""Shared data/metric utilities for Solar Filament improvement work.

Reads the single Kaggle competition zip (707 train + 180 test + COCO json).
Replicates the original date-disjoint split from solar_real.cache() exactly
(same seeds) so tune/holdout comparisons against the shipped pipeline are fair.
"""
import json, os, random, zipfile
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "PUBLIC_REPO_CONTENT"
CKPT = REPO / "results/SOLAR_REAL_TRAINED_UNET.pt"
ANNO = "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_zip():
    p = os.environ.get("SOLAR_ZIP_PATH")
    if p and Path(p).exists():
        return Path(p)
    for cand in Path(os.environ["USERPROFILE"], "Desktop").rglob("filament-segmentation-2026.zip"):
        return cand
    raise FileNotFoundError("filament-segmentation-2026.zip not found; set SOLAR_ZIP_PATH")


def load_annotations(z):
    return json.loads(z.read(ANNO))


def grouped_views(data):
    """file_name -> list of annotation-lists (one per annotator view)."""
    pics = {x["id"]: x for x in data["images"]}
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    out = defaultdict(list)
    for i, im in pics.items():
        out[Path(im["file_name"]).name].append(anns[i])
    return dict(out)


def split_names(names):
    """Exact replica of solar_real.cache() split logic."""
    dates = sorted({n[:8] for n in names})
    rng = np.random.default_rng(2026)
    rng.shuffle(dates)
    val = set(dates[: int(len(dates) * 0.18)])
    held = sorted(val)
    random.Random(4251).shuffle(held)
    hold = set(held[::2])
    out = {"train": [], "tune": [], "holdout": []}
    for n in sorted(names):
        if n[:8] not in val:
            out["train"].append(n)
        elif n[:8] in hold:
            out["holdout"].append(n)
        else:
            out["tune"].append(n)
    return out


def zip_image_index(z):
    train = {Path(n).name: n for n in z.namelist() if "/train/train_images/" in n and n.endswith(".jpeg")}
    test = {Path(n).name: n for n in z.namelist() if "/test/test_images/" in n and n.endswith(".jpeg")}
    return train, test


def read_gray(z, entry):
    im = cv2.imdecode(np.frombuffer(z.read(entry), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert im is not None and im.shape == (2048, 2048), entry
    return im


def rasterize_view(anns, size):
    """Instance map at `size` from one annotator view's polygons (2048 coords)."""
    scale = size / 2048.0
    lab = np.zeros((size, size), np.uint16)
    idx = 0
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not isinstance(seg, list):
            continue
        m = np.zeros((size, size), np.uint8)
        for poly in seg:
            pts = np.asarray(poly, float).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.rint(pts * scale).astype(np.int32)], 1)
        if m.any():
            idx += 1
            lab[(m > 0) & (lab == 0)] = idx
    return lab


# ---- PQ (same definition as solar_filament/metrics.py) ----
from scipy.optimize import linear_sum_assignment


def pq_counts(gt, pred, iou_threshold=0.5):
    gt = np.asarray(gt); pred = np.asarray(pred)
    ga = [int(x) for x in np.unique(gt) if x != 0]
    pa = [int(x) for x in np.unique(pred) if x != 0]
    if not ga and not pa:
        return {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
    if not ga:
        return {"tp": 0, "fp": len(pa), "fn": 0, "sum_iou": 0.0}
    if not pa:
        return {"tp": 0, "fp": 0, "fn": len(ga), "sum_iou": 0.0}
    area_g = {x: int(np.count_nonzero(gt == x)) for x in ga}
    area_p = {x: int(np.count_nonzero(pred == x)) for x in pa}
    overlap = np.zeros((len(ga), len(pa)), float)
    gi = {v: i for i, v in enumerate(ga)}; pi = {v: i for i, v in enumerate(pa)}
    both = (gt != 0) & (pred != 0)
    if np.any(both):
        pairs, counts = np.unique(np.stack((gt[both], pred[both]), axis=1), axis=0, return_counts=True)
        for (g, p), inter in zip(pairs, counts):
            i, j = gi[int(g)], pi[int(p)]
            overlap[i, j] = inter / (area_g[int(g)] + area_p[int(p)] - inter)
    ri, ci = linear_sum_assignment(-overlap)
    valid = overlap[ri, ci] > iou_threshold
    n = int(valid.sum())
    return {"tp": n, "fp": len(pa) - n, "fn": len(ga) - n, "sum_iou": float(overlap[ri[valid], ci[valid]].sum())}


def pooled_pq(counts):
    tp = sum(x["tp"] for x in counts); fp = sum(x["fp"] for x in counts); fn = sum(x["fn"] for x in counts)
    total = sum(x["sum_iou"] for x in counts)
    denom = tp + 0.5 * (fp + fn)
    return {"pq": float(total / denom) if denom else 1.0, "tp": tp, "fp": fp, "fn": fn,
            "rq": float(tp / denom) if denom else 1.0, "sq": float(total / tp) if tp else 0.0}
