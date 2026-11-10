# Solar Filament Segmentation Challenge 2026 — reproducible real-data entry candidate

> **Status as of November 5, 2026:** All 180 organizer test images have been **independently inferred from original JPEG bytes**, and the resulting output matches the tracked `submission.csv` **byte-for-byte**. The full Jupyter notebook has been **executed without skips or errors**. A four-page ACM `acmart/sigconf` technical report has been compiled and visually inspected. **This is a locally validated candidate; a Kaggle submission of this exact CSV on November 9, 2026 scored 0.20 on the public leaderboard (not an award).**

Official challenge: https://www.kaggle.com/competitions/filament-segmentation-2026

## The three actual deliverables

| Recipient | What to provide |
| --- | --- |
| Kaggle competition submission | **`submission.csv`** (not this Git ZIP; two columns `filament_id,segmentation_rle`) |
| Public Git repository for the organizing committee | **This entire directory** excluding the original competition JPEG/annotation data; includes source, model weights, `requirements.txt`, executed notebook and scientific evidence |
| Organizers' final Google Form | Public Git link and **`report/technical_report_4pages.pdf`** plus the team details requested on the form |

**Competition deadline: November 15, 2026, 11:59 pm CDT.** A first Kaggle submission of the tracked `submission.csv` scored **0.20** (public leaderboard, November 9, 2026); this repository is the public source release for that entry. A stronger 1024px model is in training, and this repository will be updated if it beats the shipped model on the date-disjoint holdout. The organizer Google Form has not yet been submitted. Links to the current official instructions are in `docs/FINAL_SUBMISSION_CHECKLIST.md`.

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
