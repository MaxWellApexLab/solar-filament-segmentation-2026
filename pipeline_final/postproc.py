"""Instance post-processing variants for filament probability maps.

baseline_instance_map: exact replica of the shipped solar_eval.instance_map.
instance_map_v2: configurable superset — hysteresis thresholding, erosion-seed
splitting of merged components, adjustable solar-disk erosion.
"""
import cv2, numpy as np
from scipy import ndimage


def _disk_mask(gray, erode_px):
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    disk = (blur > max(10, float(np.percentile(blur, 98)) * 0.12)).astype(np.uint8)
    n, l, stats, _ = cv2.connectedComponentsWithStats(disk, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        disk = np.uint8(l == largest)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        disk = cv2.erode(disk, k)
    return disk


def _elongation_ok(roimask, min_elongation):
    yy, xx = np.where(roimask)
    if len(xx) < 4:
        return False
    if min_elongation <= 1.0:
        return True
    eig = np.linalg.eigvalsh(np.cov(np.stack([xx, yy]).astype(float)))
    aspect = float(np.sqrt(max(1e-8, eig[-1]) / max(1e-8, eig[0])))
    return aspect >= min_elongation


def baseline_instance_map(prob, gray, threshold=0.4, min_area=70, min_elongation=1.2):
    disk = _disk_mask(gray, 11)
    binary = np.uint8((prob > threshold) & (disk > 0))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    lab = np.zeros_like(cc, np.int16); count = 0
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < min_area:
            continue
        roimask = cc[y:y + h, x:x + w] == i
        if not _elongation_ok(roimask, min_elongation):
            continue
        count += 1; lab[y:y + h, x:x + w][roimask] = count
    return lab


def _split_component(mask):
    """Split one binary component at thin bridges via erosion seeds.

    Erode until the component breaks into >=2 seeds (or give up), then assign
    every component pixel to its nearest seed. Returns list of sub-masks.
    """
    m = mask.astype(np.uint8)
    for it in (1, 2, 3):
        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=it)
        n, seeds, stats, _ = cv2.connectedComponentsWithStats(er, 8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]
        if len(keep) >= 2:
            seed_lab = np.zeros_like(seeds)
            for new, old in enumerate(keep, 1):
                seed_lab[seeds == old] = new
            _, (iy, ix) = ndimage.distance_transform_edt(seed_lab == 0, return_indices=True)
            grown = seed_lab[iy, ix] * m
            return [grown == k for k in range(1, len(keep) + 1)]
        if n <= 1:
            break
    return [mask.astype(bool)]


def instance_map_v2(prob, gray, threshold=0.4, min_area=40, min_elongation=1.2,
                    lo_threshold=None, split=False, split_min_area=400,
                    disk_erode=11, close=True):
    disk = _disk_mask(gray, disk_erode)
    binary = np.uint8((prob > threshold) & (disk > 0))
    if lo_threshold is not None and lo_threshold < threshold:
        support = np.uint8((prob > lo_threshold) & (disk > 0))
        n, cc = cv2.connectedComponents(support, 8)
        has_core = np.zeros(n, bool)
        core_ids = np.unique(cc[binary > 0])
        has_core[core_ids] = True
        has_core[0] = False
        binary = np.uint8(has_core[cc])
    if close:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    lab = np.zeros(binary.shape, np.int32); count = 0
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < min_area:
            continue
        roimask = cc[y:y + h, x:x + w] == i
        pieces = _split_component(roimask) if (split and area >= split_min_area) else [roimask]
        roi = lab[y:y + h, x:x + w]
        for piece in pieces:
            if piece.sum() < min_area:
                continue
            if not _elongation_ok(piece, min_elongation):
                continue
            count += 1; roi[piece] = count
    return lab.astype(np.int16)
