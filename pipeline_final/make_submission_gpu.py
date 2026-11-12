"""GPU submission generation for a DeepUNet checkpoint (remote GPU hosts).

Same math as make_submission.py (1024px 4-flip TTA -> instance_map_v2 ->
nearest-upscale labels with native-grid prob clip -> COCO compressed RLE),
but self-contained besides common/postproc/models so it can run in the
minimal solar_v2 folder staged on sky3/maxer. Set SOLAR_ZIP_PATH to the zip.
"""
import argparse, csv, hashlib, json, sys, zipfile
from pathlib import Path
import cv2, numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import find_zip, zip_image_index, read_gray
from postproc import instance_map_v2
from models import build_model_from_ckpt


# --- COCO compressed RLE (same wire format as solar_filament.coco_rle) ---
def counts_to_string(counts):
    out = []
    for i, v in enumerate(counts):
        x = int(v)
        if i > 2:
            x -= int(counts[i - 2])
        while True:
            c = x & 0x1f
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            c += 48
            out.append(chr(c))
            if not more:
                break
    return "".join(out)


def string_to_counts(s):
    counts = []; i = 0
    while i < len(s):
        x = 0; k = 0
        while True:
            c = ord(s[i]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            i += 1; k += 1
            if not more:
                if c & 0x10:
                    x |= -(1 << (5 * k))
                break
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def sparse_rle(labels, label):
    x, y, w, h = cv2.boundingRect(np.uint8(labels == label))
    if not w or not h:
        return None
    H, W = labels.shape; intervals = []
    for col in range(x, x + w):
        yy = np.flatnonzero(labels[y:y + h, col] == label)
        if not len(yy):
            continue
        breaks = np.flatnonzero(np.diff(yy) > 1) + 1
        for run in np.split(yy, breaks):
            start = col * H + y + int(run[0]); intervals.append((start, start + len(run)))
    intervals.sort(); counts = []; pos = 0
    for start, end in intervals:
        if start == pos and len(counts) >= 2:
            counts[-1] += end - start
        else:
            counts.extend((start - pos, end - start))
        pos = end
    if pos < H * W:
        counts.append(H * W - pos)
    assert sum(counts) == H * W
    return counts_to_string(counts)


def load_net(path, dev):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    net = build_model_from_ckpt(ck); net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval(), ck


@torch.inference_mode()
def predict_tta(net, image, dev):
    img = np.ascontiguousarray(image)
    flips = [img, img[:, ::-1], img[::-1, :], img[::-1, ::-1]]
    x = np.stack([np.ascontiguousarray(f) for f in flips]).astype(np.float32)[:, None] / 255.
    p = torch.sigmoid(net(torch.from_numpy(x).to(dev))).cpu().numpy()[:, 0]
    return (p[0] + p[1][:, ::-1] + p[2][::-1, :] + p[3][::-1, ::-1]) * 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="repeat for probability-averaged ensemble")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-area", type=int, required=True)
    ap.add_argument("--min-elongation", type=float, default=1.2)
    ap.add_argument("--disk-erode", type=int, default=21)
    ap.add_argument("--edge-threshold", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on first N images (0=all)")
    args = ap.parse_args()
    edge_thr = args.edge_threshold if args.edge_threshold is not None else args.threshold

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    print("device:", dev, gpu_name or "", flush=True)
    nets = []; metas = []
    for p in args.ckpt:
        n, m = load_net(Path(p), dev)
        nets.append(n); metas.append(m)
        print(f"ckpt {Path(p).name} epoch={m.get('epoch')} base={m.get('base')}", flush=True)

    zpath = find_zip(); print("zip:", zpath, flush=True)
    rows = 0; empty = 0; per_image = []
    with zipfile.ZipFile(zpath) as z, open(args.out, "w", newline="") as f:
        _, test_idx = zip_image_index(z)
        names = sorted(test_idx)
        if not args.limit:
            assert len(names) == 180, len(names)
        else:
            names = names[: args.limit]
        w = csv.writer(f); w.writerow(("filament_id", "segmentation_rle"))
        for i, name in enumerate(names):
            im = read_gray(z, test_idx[name])
            small = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
            prob = np.mean([predict_tta(n, small, dev) for n in nets], axis=0).astype(np.float32)
            labels = instance_map_v2(prob, small, threshold=args.threshold,
                                     min_area=args.min_area,
                                     min_elongation=args.min_elongation,
                                     disk_erode=args.disk_erode)
            native = cv2.resize(labels, (2048, 2048), interpolation=cv2.INTER_NEAREST)
            cont = cv2.resize(prob, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            final = np.where(cont >= edge_thr, native, 0).astype(np.int32)
            k = 0
            for old in range(1, int(labels.max()) + 1):
                coded = sparse_rle(final, old)
                if coded is None:
                    continue
                k += 1
                w.writerow((Path(name).stem + "_" + str(k), coded))
            rows += k; per_image.append(k); empty += int(k == 0)
            if (i + 1) % 30 == 0 or i + 1 == len(names):
                print(f"test {i+1}/{len(names)} rows={rows}", flush=True)

    ids = set(); decoded = 0
    with open(args.out, newline="") as f:
        r = csv.DictReader(f)
        assert r.fieldnames == ["filament_id", "segmentation_rle"], r.fieldnames
        for j, row in enumerate(r):
            assert row["filament_id"] not in ids
            ids.add(row["filament_id"])
            counts = string_to_counts(row["segmentation_rle"])
            assert sum(counts) == 2048 * 2048
            if j % 37 == 0:
                area = sum(counts[1::2]); assert area > 0
                decoded += 1
    audit = {"file": Path(args.out).name, "rows": rows, "images": len(per_image),
             "empty_images": empty, "mean_instances": float(np.mean(per_image)),
             "sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
             "ckpts": [str(p) for p in args.ckpt],
             "ckpt_epochs": [m.get("epoch") for m in metas],
             "device": str(dev), "gpu": gpu_name,
             "cfg": {"threshold": args.threshold, "min_area@1024": args.min_area,
                     "min_elongation": args.min_elongation, "disk_erode@1024": args.disk_erode,
                     "edge_threshold": edge_thr},
             "verified_rows_area_checked": decoded}
    Path(args.out + ".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
