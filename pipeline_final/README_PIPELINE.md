# Final pipeline entry (public leaderboard 0.34)

This directory contains the final scored entry: a two-stage detector +
per-instance crop refiner. The repository root still contains the earlier
semantic-model release (public 0.20-0.30) with its own verified artifacts;
this folder is the self-contained final candidate.

## Result
- Candidate: `submission_pipeline_final.csv` (1,585 rows, 180 images)
- SHA-256: `e1e95bf387caf00d7566e1ccecdb84f988851c5f3071730aba1ad7f0af5c2cbf`
- Kaggle public score: **0.34** (submission ref 58254552)
- Date-disjoint holdout PQ (native 2048, pooled over annotator views): 0.4209
- Config: detector score 0.70, mask 0.50, refiner 0.60, topology-safe flip-4 TTA

## Weights (GitHub Release `pipeline-v1`)
The detector checkpoint exceeds GitHub's 100 MB per-file limit, so both
checkpoints are release assets:
https://github.com/MaxWellApexLab/solar-filament-segmentation-2026/releases/tag/pipeline-v1

- `detector_best.pt` sha256 `8e5ac18657a78b413a27f9ea00b33afebdc1c8bfc2dc016c1ae05de65e9e67bd`
- `refiner_best.pt` sha256 `85af79df8335e6deb898e6886690e3b50323ddfd23e044a95826d06a6019bd7f`

Download both into this directory before running inference.

## Reproduce the candidate CSV
Requirements: Python with torch + torchvision (developed on torch
2.5.1+cu121 / torchvision 0.20.1+cu121), opencv-python-headless, scipy.
Set `SOLAR_ZIP_PATH` to the official competition zip file, then:

```
python detector_refiner.py submit --detector detector_best.pt --refiner refiner_best.pt --score-threshold 0.70 --mask-threshold 0.50 --refiner-threshold 0.60 --out submission_pipeline_final.csv
```

The command self-verifies every RLE row (counts sum to 2048^2, unique IDs)
and writes `submission_pipeline_final.csv.audit.json` beside the CSV.
GPU is strongly recommended; CPU works but is slow.

## Training (from competition data only)
```
python detector_refiner.py train-detector --epochs 10 --bs 2 --lr 0.005 --seed 2026 --out detector_best.pt
python detector_refiner.py train-refiner --epochs 8 --bs 16 --lr 3e-4 --seed 2026 --out refiner_best.pt
python detector_refiner.py eval --detector detector_best.pt --refiner refiner_best.pt --split holdout --score-threshold 0.70 --mask-threshold 0.50 --refiner-threshold 0.60 --out-json pipeline_eval_holdout.json
```

## Disclosure
Pretrained weights are the publicly available COCO Mask R-CNN and ImageNet
ResNet18 weights. No external labels and no full-MAGFiLO labels beyond the
competition training set were used. Full method, selection protocol, and
limitations are in `report/` (source + compiled PDF).
