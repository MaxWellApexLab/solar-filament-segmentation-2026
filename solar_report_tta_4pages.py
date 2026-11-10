"""Four-page evidence-accurate Solar Filament 2026 technical report candidate."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit,ImageReader

R=Path(__file__).resolve().parent;D=R/'results';OUT=D/'SOLAR_TTA_REAL_4PAGE_TECHNICAL_DRAFT.pdf'
W,H=letter;M=56;WIDTH=W-2*M
BG=HexColor('#102d46');ACC=HexColor('#167b85');TEXT=HexColor('#263749');FAINT=HexColor('#536e82')


def title(c,p,n):
    c.setFillColor(BG);c.rect(0,H-72,W,72,stroke=0,fill=1)
    c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica-Bold',15.5);c.drawString(M,H-33,f'{p} | {n}')
    c.setFont('Helvetica',9);c.setFillColor(HexColor('#d1e1ee'))
    c.drawString(M,H-53,'MAGFiLO 2026 | Authentic participant-provided GONG H-alpha image benchmark')
    c.setFillColor(FAINT);c.setFont('Helvetica',8)
    c.drawString(M,42,'Independent local science report; NOT an organizer score or accepted submission')
    c.drawRightString(W-M,42,f'{p} / 4')
    return H-97


def section(c,text,y):
    y-=7;c.setFont('Helvetica-Bold',11);c.setFillColor(ACC);c.drawString(M,y,text);return y-18


def paragraph(c,text,y,font=9.7,leading=13.4,indent=0):
    c.setFont('Helvetica',font);c.setFillColor(TEXT)
    for line in simpleSplit(str(text),'Helvetica',font,WIDTH-indent):
        if y<66:raise ValueError(f'Page overflow {line[:50]}')
        c.drawString(M+indent,y,line);y-=leading
    return y-9


def bullets(c,items,y):
    for i in items:y=paragraph(c,'- '+i,y,font=9.5,leading=13.5)
    return y


def summary_table(c,rows,y,cols=(210,122,122)):
    h=28
    for j,row in enumerate(rows):
        if j==0:
            c.setFillColor(HexColor('#dceef0'));c.rect(M,y-h+5,WIDTH,h+2,stroke=0,fill=1)
        else:
            c.setStrokeColor(HexColor('#dde6eb'));c.line(M,y-h+5,W-M,y-h+5)
        for k,v in enumerate(row):
            c.setFont('Helvetica-Bold' if j==0 or k==0 else 'Helvetica',9.3)
            c.setFillColor(TEXT);c.drawString(M+sum(cols[:k])+6,y-12,str(v))
        y-=h
    return y-8


def create():
    tta=json.loads((D/'SOLAR_TTA_COMPARISON.json').read_text())
    native=json.loads((D/'SOLAR_NATIVE2048_TTA_PAIRED.json').read_text())
    csvqa=json.loads((D/'SOLAR_TTA_ALL180_INDEPENDENT_QA.json').read_text())
    hist=json.loads((D/'SOLAR_TRAIN_HISTORY.json').read_text())
    c=canvas.Canvas(str(OUT),pagesize=letter,pageCompression=1)
    c.setTitle('Solar Filament 2026 - Real Data Four-Flip TTA Technical Draft')
    y=title(c,1,'Scientific objective and dataset')
    y=section(c,'Goal',y)
    y=paragraph(c,'Detect and segment individual dark chromospheric filaments in full-resolution (2048 x 2048) National Solar Observatory GONG H-alpha solar disk observations. The challenge emphasizes one-to-one object Panoptic Quality (PQ), false merges, fragmented objects and accurate filament boundaries.',y)
    y=section(c,'The participant-supplied official data',y)
    y=bullets(c,['707 different annotated GONG images: 589 in 00.zip and 118 in MAGFiLO Kaggle ZIP.',
        '1,154 separate image annotation views (8,199 polygons); views are NOT independent photographs.',
        '180 additional organizer test JPEGs have NO accessible ground truth and were NEVER used for model fitting.',
        'Dates are grouped before splitting: train 582 images; tune 63; local audit 62.'],y)
    y=section(c,'Core engineering method',y)
    y=paragraph(c,'A compact U-Net is fitted from scratch, followed by solar-disk masking, instance extraction and contrast-preserving four-flip test-time ensemble (TTA). The 512 x 512 working resolution supports CPU training; final RLE masks are resampled to 2048 x 2048 and independently verified.',y)
    if y>300:
        c.drawImage(ImageReader(str(D/'SOLAR_REAL_TUNE_ROW_ERROR_PANEL.png')),M,85,width=WIDTH,height=170,preserveAspectRatio=True)
        y=paragraph(c,'Illustration: actual MAGFiLO training data overlay; model disagreement and thin-filament errors are visible.',268,font=8.3,leading=11)
    c.showPage()
    y=title(c,2,'Architecture and reproducibility')
    y=section(c,'Training setup',y)
    y=bullets(c,['Original JPEGs downsampled by area interpolation; polygons rasterized separately for each annotator.',
                'Random 256 x 256 crop centered near positive pixels; brightness and horizontal/vertical flip augmentations.',
                'U-Net base width 8; AdamW with lr=6e-4 and weight decay=1e-4; weighted BCE plus soft Dice.',
                '10 full CPU epochs; checkpoint selected by tuning loss, not by Kaggle test imagery.',
                'No external weights, no test labels, no external training images used.'],y)
    y=section(c,'Four-flip ensemble and instance conversion',y)
    y=paragraph(c,'For a single image, compute sigmoid maps for the original, horizontal flip, vertical flip and both flips. Align each result back to original orientation and average the four probabilities. Use one tune-selected instance threshold, min-area and elongation filter on both baseline and TTA to isolate the ensemble effect.',y)
    y=section(c,'Multi-rater metric',y)
    y=paragraph(c,'For each distinct rater view, match instance pairs one-to-one by IoU > 0.5. Pool TP, FP, FN and matched IoU across views. PQ = sum(matched IoU) / [TP + 0.5(FP + FN)]. Never average independent per-image PQ scores as a substitute for pooled-count PQ.',y)
    y=paragraph(c,f"Ten epochs finished. First tuning loss {hist['history'][0]['tune_loss']:.4f}; best {hist['best_tune_loss']:.4f}. Model checkpoint, optimizer parameters, exact date split, ZIP source SHA-256 and test CSV QA are packaged.",y)
    if y>300:c.drawImage(ImageReader(str(D/'SOLAR_10EPOCH_REAL_TRAINING_CURVE.png')),M,82,width=WIDTH,height=174,preserveAspectRatio=True)
    c.showPage()
    y=title(c,3,'Measured results: 512px and native 2048px')
    y=section(c,'Ablation: baseline versus four-flip TTA',y)
    y=summary_table(c,[['Local multi-rater PQ','Baseline','4-flip TTA'],
                       ['512px tune / 63 images','0.3263','0.3408'],
                       ['512px audit / 62 images','0.3098','0.3281'],
                       ['2048px audit / 62 images',f"{native['baseline']['pq']:.4f}",f"{native['four_flip_tta']['pq']:.4f}"],
                       ['2048px successful matches','247','272'],
                       ['2048px false detections','296','279'],
                       ['2048px missed objects','473','448']],y)
    y=section(c,'Paired uncertainty analysis',y)
    low,high=native['paired_bootstrap_95_percentile_ci']
    y=paragraph(c,f"On 62 date-disjoint images and {native['annotator_views']} annotation views, TTA minus baseline native 2048-pixel PQ is {native['tta_minus_baseline_pq']:+.4f}; 95% paired image-level bootstrap interval [{low:+.4f}, {high:+.4f}] from 2,000 image-group resamples. The same objects/annotation views and segmentation thresholds were used for both methods.",y)
    y=section(c,'Deployment artifact (unlabeled 180-image Kaggle test)',y)
    y=paragraph(c,f"Generated {csvqa['valid_rows']} unique candidate filament rows across {csvqa['expected_test_images']} official test images; {csvqa['empty_prediction_images']} images yield no prediction. For ALL rows, column-major COCO-RLE decodes to exactly 2048 x 2048 pixels and IDs map to official test filenames. Independently decoded {csvqa['actual_decoded_2048_masks']} 2048-pixel masks; SHA-256 verified.",y)
    y=section(c,'Threats to scientific validity',y)
    y=bullets(c,['The model operates at 512px, which removes thin strands and degrades true native-resolution geometry.',
                'Local audit dates were previously inspected in the project; this is NOT an untouched blind test.',
                'Bootstrap assumes approximate image independence; time correlation may narrow the confidence interval.',
                'Official private-score computation and independent model robustness are not verified.'],y)
    c.showPage()
    y=title(c,4,'Submission protocol and evidence boundaries')
    y=section(c,'Reproduce this research',y)
    y=bullets(c,['Keep the original two organizer archives intact. Set SOLAR_ZIP1 and SOLAR_ZIP2 if paths differ.',
                'python -c "import solar_real;solar_real.cache()" (creates date-separated view-specific cache).',
                'python solar_tta.py (tune-only selection, independent audit; rebuild 180-test inference).',
                'python solar_native2048_tta_paired.py (original polygon PQ and paired bootstrap).',
                'python solar_verify_tta_submission.py (all 1,013 RLE strings, source image IDs, SHA-256).'],y)
    y=section(c,'Delivered artifacts',y)
    y=paragraph(c,'SOLAR_180_TEST_TTA_ALTERNATE.csv (180 test images); trained U-Net checkpoint; TTA selection report; full-resolution paired native polygon PQ; bootstrap uncertainty; full submission CSV audit; source code; manifest; this four-page PDF. The participant-provided 670 MB+ image archives are NOT redistributed.',y)
    y=section(c,'Formal submission remains a separate action',y)
    y=bullets(c,['Candidate CSV is machine-valid but has NOT been submitted to Kaggle or officially scored.',
                'Participant must verify account, award eligibility, correct original files and rule compliance.',
                'Public Git repository URL, code license and final personal/team attribution need completion by participant.',
                'AI-assisted coding and report drafting must be disclosed as required by the competition rules.'],y)
    y=section(c,'Organizer and provenance references',y)
    y=paragraph(c,'NSO GONG H-alpha: https://nso.edu/data/nisp-data/h-alpha/ ; official competition: https://www.kaggle.com/competitions/filament-segmentation-2026 ; canonical COCO RLE: https://github.com/cocodataset/cocoapi.',y,font=8.9)
    y=section(c,'Conclusion',y)
    y=paragraph(c,'Four-flip ensemble improves local native-resolution segmentation on matched reference images, but false positives, misses and edge fidelity remain substantial. This is a reproducible real-data engineering candidate, NOT a claim of winning accuracy or a published award.',y)
    c.showPage();c.save()
    return OUT

if __name__=='__main__':print(create())
