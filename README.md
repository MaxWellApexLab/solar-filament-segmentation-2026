# Solar Filament Segmentation Challenge 2026 — reproducible real-data entry candidate

> **Status as of November 11, 2026:** The final entry is the two-stage detector+refiner pipeline in [`pipeline_final/`](pipeline_final/README_PIPELINE.md), which scored **0.34 on the public leaderboard** (submission ref 58254552; date-disjoint native-resolution holdout PQ 0.4209). The earlier semantic-model release documented below remains fully preserved and verified; its tracked `submission.csv` scored 0.20, and later semantic candidates reached 0.26--0.30. Public scores are not awards, and the private score is not yet known.

Official challenge: https://www.kaggle.com/competitions/filament-segmentation-2026

## The three actual deliverables

| Recipient | What to provide |
| --- | --- |
| Kaggle competition submission | **`submission.csv`** (not this Git ZIP; two columns `filament_id,segmentation_rle`) |
| Public Git repository for the organizing committee | **This entire directory** excluding the original competition JPEG/annotation data; includes source, model weights, `requirements.txt`, executed notebook and scientific evidence |
| Organizers' final Google Form | Public Git link and **`report/technical_report_4pages.pdf`** plus the team details requested on the form |

**Competition deadline: November 15, 2026, 11:59 pm CDT.** Six candidates were submitted through the Kaggle CLI (public scores 0.20, 0.26, 0.30, 0.29, 0.29, and the final pipeline at **0.34**); every candidate was selected by date-disjoint holdout PQ before submission, never by leaderboard feedback alone. This repository is the public source release for the final pipeline entry; the organizer Google Form and final-submission selection are completed by the participant. Links to the current official instructions are in `docs/FINAL_SUBMISSION_CHECKLIST.md`.

## Final pipeline (2026-11-11, public 0.34)

The scored final candidate is a Mask R-CNN (ResNet50 FPN v2, COCO-pretrained) detector at 1024px plus a ResNet18 U-Net refiner on native-resolution per-instance crops with topology-safe four-flip TTA (detector 0.70 / mask 0.50 / refiner 0.60). Native-resolution holdout PQ is **0.4209** (detector only 0.3421), pooled over all annotator views. Code, candidate CSV + audit, evaluation JSONs, and the four-page pipeline report are in [`pipeline_final/`](pipeline_final/README_PIPELINE.md); checkpoints are GitHub Release assets ([pipeline-v1](https://github.com/MaxWellApexLab/solar-filament-segmentation-2026/releases/tag/pipeline-v1)) because the detector exceeds GitHub's 100 MB file limit. The semantic-model sections below remain as the honest development record and ablation ladder for this pipeline.

## Verified engineering results

The participant supplied both parts of the **same authorized** MAGFiLO 2026 Kaggle dataset: `00.zip` (589 labeled GONG H-alpha images) and `MAGFiLO_1.0_Kaggle_2026.zip` (118 labeled images, the 180 unlabeled test images, and a shared train-only annotation JSON). There are 707 distinct labeled physical photos, 1,154 annotation views, and 8,199 polygon instances. Data splits by date: fit 582; tuning 63; local audit 62.

The included **real trained** CPU U-Net (`results/SOLAR_REAL_TRAINED_UNET.pt`) predicts filament probability on 512x512 images. Four-flip inference and tune-selected connected components generate objects; edge-refined masks are produced at 2048x2048. On 62 **date-separated but previously inspected training-audit** photos with 93 annotator views, original-resolution pooled PQ:

| Method | Local 2048px PQ |
| --- | ---: |
| Unaugmented model | 0.2395 |
| Four-flip TTA | 0.262893 |
| Four-flip TTA + native-grid boundary clip | **0.273115** |

The edge-refinement change is **+0.010221** PQ, with an image-level paired bootstrap 95% percentile interval [+0.001687,+0.018674] based on 2,000 resamples. This is **not a private leaderboard estimate**. Per-photo, 43 improve, 12 deteriorate and 7 remain unchanged. Limitations are detailed in the PDF.

### Exact full-test inference verification

- **180/180** authentic official competition test JPEGs processed from scratch (no predicted mask seeds).
- **1,013** unique object rows, with 6 images having no accepted object.
- All output RLE count vectors have **2048 x 2048** pixel area, object IDs are unique and consecutive within each source image, and 30 full-size masks were independently decoded.
- Final candidate digest (also reproduced with an extracted Kaggle directory):

  `e0e55cf4eb383769c17b742d556d619859307b82a4267792ad2345698126d9ea`

- `notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb` documents a fully successful real-data run. `results/SOLAR_NOTEBOOK_EXECUTION_RECEIPT.json` has a model+CSV provenance proof.

## Reproduce on Kaggle or locally

**Required for prediction:** the code in this repository, its bundled checkpoint, and **only the original 180 unlabelled test JPEGs supplied by the competition**. No external checkpoint download, API keys, training labels, or model-training service is required to reproduce the exact CSV.

Use a supported Python environment and install `pip install -r requirements.txt` (all packages pinned; tested here with PyTorch `2.10.0+cpu`, matching the canonical `torch==2.10.0` dependency). For a Kaggle notebook, attach the original official competition dataset under `/kaggle/input/`.

```bash
# Local ZIP version (your already-supplied competition archive):
export SOLAR_ZIP2=/path/to/MAGFiLO_1.0_Kaggle_2026.zip
python reproduce_submission.py --test-zip "$SOLAR_ZIP2" \
  --shards _repro_shards \
  --output results/SOLAR_REPRODUCED_FROM_OFFICIAL_JPEGS.csv

# Or original extracted Kaggle data directory:
python reproduce_submission.py --test-zip /kaggle/input/ \
  --shards _repro_shards \
  --output results/SOLAR_REPRODUCED_FROM_OFFICIAL_JPEGS.csv

# Then verify the entire output against all 180 official image IDs:
python verify_candidate.py results/SOLAR_REPRODUCED_FROM_OFFICIAL_JPEGS.csv "$SOLAR_ZIP2"

# Or run the complete notebook, including actual model forward passes:
python scripts/execute_notebook.py

# Preflight the final PDF, hashes, model, code and notebook:
python scripts/final_preflight.py
pytest -q tests
```

`reproduce_submission.py` creates per-image shards **atomically** and validates checkpoint SHA-256 plus JPEG CRC before resuming. A complete new run was performed in the reference environment, and a separate executable notebook run confirmed the entire 180-image result. A fresh clone with an empty `_repro_shards` **actually computes all 180 predictions**; it never silently loads a precomputed inference CSV in place of model predictions.

### Optional retraining and independent evaluation

A ten-epoch retrain requires **both** authorized Kaggle training ZIP fragments; their original bytes are not redistributed in this repository:

```bash
export SOLAR_ZIP1=/path/to/00.zip
export SOLAR_ZIP2=/path/to/MAGFiLO_1.0_Kaggle_2026.zip
python -c 'import solar_real;solar_real.cache();solar_real.train(epochs=10)'
python solar_tta.py
python edge_native_validation.py batch --evidence-dir _original_polygon_evidence --first 0 --count 10
# Repeat the batch over all 62 local audit photos before assembly.
python edge_native_validation.py assemble --evidence-dir _original_polygon_evidence
```

**Do not retrain unless you intend to reselect a checkpoint:** refitting may produce different weights and should not be represented as byte-for-byte reproduction of the existing candidate. Local evaluation has no access to organizer-hidden test masks.

## Repository structure

`submission.csv` (direct Kaggle upload), `results/SOLAR_REAL_TRAINED_UNET.pt` (genuine trained checkpoint), `reproduce_submission.py` (atomic full-test inference), `solar_source.py` (ZIP/extracted test-file reader), `verify_candidate.py` (independent mask validation), `notebooks/REPRODUCE_SOLAR.ipynb` and `notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb` (the complete working pipeline), `report/main.tex`, `report/technical_report_4pages.pdf` (4-page ACM formatted technical draft), `tests/` (checkpoints, binary-mask encoding and reproduction tests), and `docs/FINAL_SUBMISSION_CHECKLIST.md` (the actual publication actions).

## Important boundaries

- The organizer public leaderboard score for this exact CSV is **0.20**; private/final scoring is not known. A valid CSV and strong reproducibility are not prizes.
- The 512px model misses weak strands and can over-merge branches; original-resolution local PQ is only 0.2731. This is not proof of award competitiveness.
- Team name, actual author affiliations and approval of the report's exact Overleaf template formatting must be reviewed by the participant before final submission. The public `acmart/sigconf` source is included; the organizer's private Overleaf read-only template could not be independently downloaded in this environment.
- The source code MIT license applies to this project's code subject to participant approval; it does not relicense the competition data or trained-model derivation rights. See `DATA_AND_MODEL_RIGHTS.md`.
- AI assistance was used for software and documentation. Under the official rules, prize eligibility also requires the submitting participant to be able to explain and defend the methodology.

Relevant sources: [competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates), [official rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules), [MAGFiLO paper](https://doi.org/10.1038/s41597-024-03876-y), and [official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
