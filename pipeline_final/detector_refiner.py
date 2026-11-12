"""Two-stage instance pipeline: Mask R-CNN detector + per-instance crop refiner.

Independent implementation using torchvision's standard Mask R-CNN and a
ResNet18-encoder U-Net refiner. Data protocol matches the rest of IMPROVE_V2:
date-disjoint split from common.split_names, one selected annotator view per
physical observation for training, pooled all-view PQ for evaluation.
Subcommands: train-detector | train-refiner | eval | submit
"""
import argparse, csv, hashlib, json, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (find_zip, load_annotations, grouped_views, split_names,
                    zip_image_index, read_gray, rasterize_view,
                    pq_counts, pooled_pq)


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


# --- data helpers ---
def select_view(views_for_name):
    """Annotator view with most instances (then most area): one training target
    set per physical observation."""
    return max(views_for_name,
               key=lambda anns: (len(anns), sum(float(a.get("area", 0)) for a in anns)))


def load_views():
    zpath = find_zip()
    with zipfile.ZipFile(zpath) as z:
        data = load_annotations(z)
        views = grouped_views(data)
        train_idx, test_idx = zip_image_index(z)
    return zpath, views, train_idx, test_idx, split_names(views.keys())


def load_mem1024(names):
    """name -> (uint8 image 1024, selected-view instance label map 1024)."""
    zpath, views, train_idx, _, _ = load_views()
    mem = {}
    with zipfile.ZipFile(zpath) as z:
        for i, name in enumerate(names):
            im = read_gray(z, train_idx[name])
            small = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
            lab = rasterize_view(select_view(views[name]), 1024)
            mem[name] = (small, lab.astype(np.int32))
            if (i + 1) % 150 == 0:
                print(f"mem1024 {i+1}/{len(names)}", flush=True)
    return mem


def masks_from_labels(lab):
    return [lab == i for i in range(1, int(lab.max()) + 1) if np.any(lab == i)]


def boxes_of(masks):
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def merge_overlaps(masks):
    """Union groups of positively-overlapping bool masks (bbox-prefiltered)."""
    masks = [np.asarray(m, dtype=bool) for m in masks if np.any(m)]
    if len(masks) < 2:
        return masks
    boxes = boxes_of(masks)
    parent = list(range(len(masks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            x0 = max(int(boxes[i, 0]), int(boxes[j, 0]))
            y0 = max(int(boxes[i, 1]), int(boxes[j, 1]))
            x1 = min(int(boxes[i, 2]), int(boxes[j, 2]))
            y1 = min(int(boxes[i, 3]), int(boxes[j, 3]))
            if x1 <= x0 or y1 <= y0:
                continue
            if np.any(masks[i][y0:y1, x0:x1] & masks[j][y0:y1, x0:x1]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    return [np.logical_or.reduce([masks[i] for i in g]) for g in groups.values()]


def assign_labels(masks, shape):
    lab = np.zeros(shape, np.int32)
    for i, m in enumerate(masks, 1):
        lab[np.asarray(m, dtype=bool) & (lab == 0)] = i
    return lab


def square_bounds(mask, context=1.8, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def photometric(image, rng):
    valid = image > 0.01
    if not np.any(valid):
        return image
    if rng.random() < 0.70:
        image[valid] = image[valid] ** float(rng.uniform(0.80, 1.25))
    if rng.random() < 0.70:
        mean = image[valid].mean()
        image[valid] = ((image[valid] - mean) * float(rng.uniform(0.85, 1.15))
                        + mean) * float(rng.uniform(0.90, 1.10))
    if rng.random() < 0.15:
        blurred = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.30, 0.80)))
        image[valid] = blurred[valid]
    if rng.random() < 0.25:
        image[valid] += rng.normal(0, float(rng.uniform(0.003, 0.015)), int(valid.sum()))
    return np.clip(image, 0, 1)


class DetectorDataset(Dataset):
    def __init__(self, names, mem, augment=False, seed=2026):
        self.names = names; self.mem = mem; self.augment = augment; self.seed = seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        im, lab = self.mem[name]
        image = im.astype(np.float32) / 255.
        masks = masks_from_labels(lab)
        if self.augment:
            rng = np.random.default_rng(self.seed + 7919 * index
                                          + 104729 * int(np.random.randint(1 << 30)))
            k = int(rng.integers(4))
            if k:
                image = np.rot90(image, k)
                masks = [np.rot90(m, k) for m in masks]
            if rng.random() < 0.5:
                image = np.fliplr(image)
                masks = [np.fliplr(m) for m in masks]
            image = photometric(np.ascontiguousarray(image), rng)
        masks = [np.ascontiguousarray(m) for m in masks if np.any(m)]
        boxes = boxes_of(masks)
        if len(masks):
            mask_t = torch.from_numpy(np.stack(masks).astype(np.uint8))
        else:
            mask_t = torch.zeros((0, 1024, 1024), dtype=torch.uint8)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(masks), dtype=torch.int64),
            "masks": mask_t,
            "image_id": torch.tensor([index]),
            "area": torch.as_tensor(mask_t.sum((1, 2)) if len(masks) else [],
                                     dtype=torch.float32),
            "iscrowd": torch.zeros(len(masks), dtype=torch.int64),
        }
        image_t = torch.from_numpy(
            np.ascontiguousarray(np.repeat(image[None], 3, axis=0)))
        return image_t, target, name


def collate(batch):
    return tuple(zip(*batch))


def build_detector(pretrained=True):
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (1024,), 1024
    return model


@torch.inference_mode()
def detector_masks(model, im1024_uint8, dev, score_thr=0.60, mask_thr=0.50):
    x = torch.from_numpy(
        np.repeat((im1024_uint8.astype(np.float32) / 255.)[None], 3, axis=0)).to(dev)
    out = model([x])[0]
    keep = out["scores"] >= score_thr
    masks = (out["masks"][keep, 0] >= mask_thr).cpu().numpy().astype(bool)
    scores = out["scores"][keep].cpu().numpy()
    return merge_overlaps(list(masks)), scores


@torch.inference_mode()
def tune_detector_pq(model, names, mem, dev, score_thr=0.60, mask_thr=0.50):
    model.eval(); counts = []
    for name in names:
        im, lab = mem[name]
        masks, _ = detector_masks(model, im, dev, score_thr, mask_thr)
        counts.append(pq_counts(lab, assign_labels(masks, lab.shape)))
    model.train()
    return pooled_pq(counts)


def train_detector(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    _, _, _, _, splits = load_views()
    mem = load_mem1024(splits["train"] + splits["tune"])
    print(f"mem1024 loaded {len(mem)}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_detector(pretrained=True).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M MaskRCNN-R50FPNv2", flush=True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    milestones = sorted({max(1, round(args.epochs * 0.70)),
                         max(2, round(args.epochs * 0.90))})
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    train_names = [n for s in args.train_splits for n in splits[s]]
    final_fit = "tune" in args.train_splits
    print(f"train splits {args.train_splits}: {len(train_names)} images final_fit={final_fit}", flush=True)
    ds = DetectorDataset(train_names, mem, augment=True, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0,
                    collate_fn=collate)
    out = HERE / args.out
    hist_path = HERE / (args.out.replace(".pt", "_history.json"))
    hist = []; best = -1.0
    for epoch in range(args.epochs):
        t0 = time.time(); model.train(); losses = []
        for si, (images, targets, _) in enumerate(dl):
            images = [im.to(dev, non_blocking=True) for im in images]
            targets = [{k: v.to(dev, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=(dev.type == "cuda")):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach()))
            if (si + 1) % 100 == 0:
                print(f"epoch {epoch+1} step {si+1}/{len(dl)} loss {np.mean(losses[-100:]):.4f}", flush=True)
        sched.step()
        r = tune_detector_pq(model, splits["tune"], mem, dev,
                             args.score_threshold, args.mask_threshold)
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "tune_pq_selected_view": r["pq"], "tune_rq": r["rq"],
               "tune_sq": r["sq"], "seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"EPOCH {epoch+1} loss {rec['train_loss']:.4f} tunePQ {r['pq']:.4f} "
              f"(RQ {r['rq']:.3f} SQ {r['sq']:.3f}) {rec['seconds']:.0f}s", flush=True)
        if (not final_fit and r["pq"] > best) or (final_fit and epoch == args.epochs - 1):
            best = r["pq"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "tune_pq": best, "arch": "MaskRCNN_ResNet50_FPN_V2"}, out)
            print(f"  saved {'final' if final_fit else 'best'} -> {out.name}", flush=True)
        hist_path.write_text(json.dumps({"history": hist, "best_tune_pq": best}, indent=2))
    print("DETECTOR TRAINING COMPLETE best tune PQ", best, flush=True)


# --- per-instance crop refiner ---
def build_crops(names):
    """Native-resolution per-instance crops (selected annotator view)."""
    zpath, views, train_idx, _, _ = load_views()
    images, masks = [], []
    with zipfile.ZipFile(zpath) as z:
        for ni, name in enumerate(names):
            im = read_gray(z, train_idx[name])
            lab = rasterize_view(select_view(views[name]), 2048)
            for inst in range(1, int(lab.max()) + 1):
                m = lab == inst
                if not np.any(m):
                    continue
                x0, y0, x1, y1 = square_bounds(m)
                images.append(cv2.resize(im[y0:y1, x0:x1], (256, 256),
                                         interpolation=cv2.INTER_LINEAR))
                masks.append(cv2.resize(m[y0:y1, x0:x1].astype(np.uint8),
                                        (256, 256), interpolation=cv2.INTER_NEAREST))
            if (ni + 1) % 100 == 0:
                print(f"crops {ni+1}/{len(names)} -> {len(images)}", flush=True)
    return np.stack(images), np.stack(masks)


class CropDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images, self.masks, self.augment = images, masks, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.float32) / 255.
        mask = self.masks[index].astype(np.float32)
        if self.augment:
            k = int(np.random.randint(4))
            if k:
                image = np.rot90(image, k); mask = np.rot90(mask, k)
            if np.random.rand() < 0.5:
                image = np.fliplr(image); mask = np.fliplr(mask)
            if np.random.rand() < 0.5:
                image = image ** float(np.random.uniform(0.85, 1.20))
            if np.random.rand() < 0.5:
                image = np.clip(image * float(np.random.uniform(0.90, 1.10)), 0, 1)
        x = np.repeat(np.ascontiguousarray(image)[None], 3, axis=0)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(mask[None])))


class ConvBlock(nn.Sequential):
    def __init__(self, ni, no):
        super().__init__(
            nn.Conv2d(ni, no, 3, padding=1, bias=False), nn.BatchNorm2d(no),
            nn.ReLU(inplace=True),
            nn.Conv2d(no, no, 3, padding=1, bias=False), nn.BatchNorm2d(no),
            nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    """ResNet18-encoder U-Net for 256x256 instance crops; 1ch 0..1 in."""
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        enc = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2
        self.layer3, self.layer4 = enc.layer3, enc.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.img_mean) / self.img_std
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        y = self.dec4(torch.cat([self.up4(x4), x3], 1))
        y = self.dec3(torch.cat([self.up3(y), x2], 1))
        y = self.dec2(torch.cat([self.up2(y), x1], 1))
        y = self.dec1(torch.cat([self.up1(y), x0], 1))
        return self.head(self.up0(y))


def soft_dice_loss(logits, targets):
    p = logits.sigmoid()
    inter = (p * targets).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def train_refiner(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    _, _, _, _, splits = load_views()
    train_names = [n for s in args.train_splits for n in splits[s]]
    tr_im, tr_mk = build_crops(train_names)
    va_im = va_mk = None
    if "tune" not in args.train_splits:
        va_im, va_mk = build_crops(splits["tune"])
    print(f"crops train {len(tr_im)} val {0 if va_im is None else len(va_im)}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = CropUNet(pretrained=True).to(dev)
    print(f"params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M CropUNet-R18", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    dl = DataLoader(CropDataset(tr_im, tr_mk, augment=True),
                    batch_size=args.bs, shuffle=True, num_workers=0)
    vl = (DataLoader(CropDataset(va_im, va_mk, augment=False),
                     batch_size=args.bs * 2, shuffle=False, num_workers=0)
          if va_im is not None else None)
    out = HERE / args.out
    hist_path = HERE / (args.out.replace(".pt", "_history.json"))
    hist = []; best = -1.0
    for epoch in range(args.epochs):
        t0 = time.time(); net.train(); losses = []
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=(dev.type == "cuda")):
                logits = net(x)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, y) \
                    + 0.5 * soft_dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        val_dice = None
        if vl is not None:
            net.eval(); tot = 0.0; n = 0
            with torch.inference_mode():
                for x, y in vl:
                    yb = y.to(dev).bool()
                    pred = net(x.to(dev)).sigmoid() >= 0.65
                    inter = (pred & yb).sum((1, 2, 3)).float()
                    den = pred.sum((1, 2, 3)) + yb.sum((1, 2, 3))
                    tot += ((2 * inter + 1e-7) / (den + 1e-7)).sum().item(); n += len(x)
            val_dice = tot / n
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "val_dice": val_dice, "seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"EPOCH {epoch+1} loss {rec['train_loss']:.4f} "
              f"valDice {val_dice if val_dice is None else round(val_dice,4)} "
              f"{rec['seconds']:.0f}s", flush=True)
        score = val_dice if val_dice is not None else -rec["train_loss"]
        if score > best:
            best = score
            torch.save({"model": net.state_dict(), "epoch": epoch + 1,
                        "val_dice": val_dice, "arch": "CropUNet_R18"}, out)
            print(f"  saved best -> {out.name}", flush=True)
        hist_path.write_text(json.dumps({"history": hist, "best": best}, indent=2))
    print("REFINER TRAINING COMPLETE best", best, flush=True)


# --- two-stage inference ---
TTA_TRANSFORMS = ("identity", "hflip", "vflip", "hvflip")


def flip_batch(a, transform):
    if transform == "identity":
        return np.ascontiguousarray(a)
    if transform == "hflip":
        return np.ascontiguousarray(a[:, :, ::-1])
    if transform == "vflip":
        return np.ascontiguousarray(a[:, ::-1, :])
    if transform == "hvflip":
        return np.ascontiguousarray(a[:, ::-1, ::-1])
    raise ValueError(transform)


@torch.inference_mode()
def run_refiner(refiner, crops, dev, bs=32):
    outs = []
    for s in range(0, len(crops), bs):
        chunk = np.ascontiguousarray(crops[s:s + bs])
        x = torch.from_numpy(chunk.astype(np.float32) / 255.).unsqueeze(1).to(dev)
        outs.append(refiner(x).sigmoid().cpu().numpy()[:, 0])
    return np.concatenate(outs).astype(np.float32)


def mask_dice(a, b):
    inter = int((a & b).sum())
    return 2 * inter / max(int(a.sum()) + int(b.sum()), 1)


def choose_topology_safe(single_p, tta_p, thr):
    single = single_p >= thr; tta = tta_p >= thr
    sa, ta = int(single.sum()), int(tta.sum())
    if sa == 0 or ta == 0:
        return single_p
    ratio = ta / max(sa, 1)
    if ratio < 0.60 or ratio > 1.70:
        return single_p
    sc = int(cv2.connectedComponents(single.astype(np.uint8), connectivity=8)[0] - 1)
    tc = int(cv2.connectedComponents(tta.astype(np.uint8), connectivity=8)[0] - 1)
    if tc > max(sc + 1, 2):
        return single_p
    if sa >= 32 and mask_dice(single, tta) < 0.62:
        return single_p
    return tta_p


def refine_probs(refiner, crops, dev, use_tta, thr):
    single = run_refiner(refiner, crops, dev)
    if not use_tta:
        return single
    acc = single.astype(np.float64).copy()
    for t in TTA_TRANSFORMS[1:]:
        p = run_refiner(refiner, flip_batch(crops, t), dev)
        acc += flip_batch(p, t).astype(np.float64)
    mean = (acc / len(TTA_TRANSFORMS)).astype(np.float32)
    return np.stack([choose_topology_safe(s, m, thr)
                     for s, m in zip(single, mean)])


@torch.inference_mode()
def pipeline_masks(gray2048, detector, refiner, dev, score_thr=0.70,
                   mask_thr=0.50, refiner_thr=0.70, use_tta=True):
    small = cv2.resize(gray2048, (1024, 1024), interpolation=cv2.INTER_AREA)
    det_masks, _ = detector_masks(detector, small, dev, score_thr, mask_thr)
    height, width = gray2048.shape
    full_masks = []
    for m in det_masks:
        full_masks.append(cv2.resize(m.astype(np.uint8), (width, height),
                                     interpolation=cv2.INTER_NEAREST).astype(bool))
    if refiner is None or not full_masks:
        return full_masks
    tasks = []
    for full in full_masks:
        x0, y0, x1, y1 = square_bounds(full)
        crop = cv2.resize(gray2048[y0:y1, x0:x1], (256, 256),
                           interpolation=cv2.INTER_LINEAR)
        tasks.append({"bounds": (x0, y0, x1, y1), "crop": crop, "fallback": full})
    probs = refine_probs(refiner, np.stack([t["crop"] for t in tasks]), dev,
                         use_tta, refiner_thr)
    refined = []
    for task, prob in zip(tasks, probs):
        x0, y0, x1, y1 = task["bounds"]
        crop_mask = cv2.resize(prob, (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_LINEAR) >= refiner_thr
        if np.any(crop_mask):
            full = np.zeros((height, width), bool)
            full[y0:y1, x0:x1] = crop_mask
            refined.append(full)
        else:
            refined.append(task["fallback"])
    return merge_overlaps(refined)


def load_pipeline(args, dev):
    detector = build_detector(pretrained=False).to(dev)
    ck = torch.load(args.detector, map_location=dev, weights_only=False)
    detector.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    detector.eval()
    refiner = None
    if getattr(args, "refiner", None):
        refiner = CropUNet(pretrained=False).to(dev)
        rk = torch.load(args.refiner, map_location=dev, weights_only=False)
        refiner.load_state_dict(rk["model"] if isinstance(rk, dict) and "model" in rk else rk)
        refiner.eval()
    return detector, refiner


def eval_pipeline(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    detector, refiner = load_pipeline(args, dev)
    zpath, views, train_idx, _, splits = load_views()
    names = splits[args.split]
    counts = []
    with zipfile.ZipFile(zpath) as z:
        for i, name in enumerate(names):
            gray = read_gray(z, train_idx[name])
            masks = pipeline_masks(gray, detector, refiner, dev,
                                   args.score_threshold, args.mask_threshold,
                                   args.refiner_threshold, args.tta)
            pred_lab = assign_labels(masks, gray.shape)
            for anns in views[name]:
                gt_lab = rasterize_view(anns, 2048)
                counts.append(pq_counts(gt_lab, pred_lab))
            if (i + 1) % 10 == 0:
                print(f"eval {i+1}/{len(names)}", flush=True)
    result = pooled_pq(counts)
    out = {"split": args.split, "detector": str(args.detector),
           "refiner": str(args.refiner), "cfg": {
               "score_threshold": args.score_threshold,
               "mask_threshold": args.mask_threshold,
               "refiner_threshold": args.refiner_threshold,
               "tta": bool(args.tta)}, "pooled": result}
    (HERE / args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def submit_pipeline(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    detector, refiner = load_pipeline(args, dev)
    zpath, _, _, test_idx, _ = load_views()
    names = sorted(test_idx)
    assert len(names) == 180, len(names)
    rows = 0; empty = 0; per = []
    with zipfile.ZipFile(zpath) as z, open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(("filament_id", "segmentation_rle"))
        for i, name in enumerate(names):
            gray = read_gray(z, test_idx[name])
            masks = pipeline_masks(gray, detector, refiner, dev,
                                   args.score_threshold, args.mask_threshold,
                                   args.refiner_threshold, args.tta)
            lab = assign_labels(masks, gray.shape)
            k = 0
            for old in range(1, int(lab.max()) + 1):
                coded = sparse_rle(lab, old)
                if coded is None:
                    continue
                k += 1
                w.writerow((Path(name).stem + "_" + str(k), coded))
            rows += k; per.append(k); empty += int(k == 0)
            if (i + 1) % 30 == 0 or i + 1 == len(names):
                print(f"test {i+1}/180 rows={rows}", flush=True)
    ids = set()
    with open(args.out, newline="") as f:
        r = csv.DictReader(f)
        assert r.fieldnames == ["filament_id", "segmentation_rle"]
        for row in r:
            assert row["filament_id"] not in ids
            ids.add(row["filament_id"])
            assert sum(string_to_counts(row["segmentation_rle"])) == 2048 * 2048
    audit = {"file": Path(args.out).name, "rows": rows, "images": 180,
             "empty_images": empty, "mean_instances": float(np.mean(per)),
             "sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
             "detector": str(args.detector), "refiner": str(args.refiner),
             "device": str(dev),
             "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
             "cfg": {"score_threshold": args.score_threshold,
                     "mask_threshold": args.mask_threshold,
                     "refiner_threshold": args.refiner_threshold,
                     "tta": bool(args.tta)}}
    Path(args.out + ".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train-detector")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--score-threshold", type=float, default=0.60)
    p.add_argument("--mask-threshold", type=float, default=0.50)
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--out", default="detector_best.pt")

    p = sub.add_parser("train-refiner")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--out", default="refiner_best.pt")

    for name in ("eval", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--detector", required=True)
        p.add_argument("--refiner", default=None)
        p.add_argument("--score-threshold", type=float, default=0.70)
        p.add_argument("--mask-threshold", type=float, default=0.50)
        p.add_argument("--refiner-threshold", type=float, default=0.70)
        p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
        if name == "eval":
            p.add_argument("--split", choices=["tune", "holdout"], default="holdout")
            p.add_argument("--out-json", default="pipeline_eval.json")
        else:
            p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "train-detector":
        train_detector(args)
    elif args.cmd == "train-refiner":
        train_refiner(args)
    elif args.cmd == "eval":
        eval_pipeline(args)
    else:
        submit_pipeline(args)


if __name__ == "__main__":
    main()
