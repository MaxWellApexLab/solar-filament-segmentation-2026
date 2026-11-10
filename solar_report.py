"""Generate four-page solar genuine-annotation technical candidate; no invented score."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit, ImageReader
R=Path(__file__).resolve().parent; O=R/'results'; P=O/'SOLAR_REAL_4PAGE_TECHNICAL_REPORT.pdf'
W,H=letter

def para(c,body,y,size=10,leading=14):
 c.setFillColor(HexColor('#21334b'));c.setFont('Helvetica',size)
 for line in simpleSplit(str(body),'Helvetica',size,480):
  if y<76:
   raise RuntimeError('Page overflow, '+str(body)[:30])
  c.drawString(65,y,line)
  y-=leading
 return y-11

def head(c,n,title):
 c.setFillColor(HexColor('#10293d'));c.rect(0,727,612,65,fill=1,stroke=0)
 c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica-Bold',15);c.drawString(65,753,title)
 c.setFillColor(HexColor('#b9d5e2'));c.setFont('Helvetica',9);c.drawString(65,736,'Solar Filament Segmentation - real MAGFiLO 2026 engineering candidate')
 c.setFillColor(HexColor('#647b92'));c.setFont('Helvetica',8)
 c.drawString(65,45,'Not an official Kaggle score, award, or verified contest submission')
 c.drawRightString(547,45,f'{n}/4')
 return 702

def sub(c,title,y):
 c.setFillColor(HexColor('#176c7c'));c.setFont('Helvetica-Bold',12)
 c.drawString(65,y,title);return y-22

def bul(c,items,y):
 for item in items:y=para(c,'- '+item,y,9.8,13)
 return y

def generate():
 hist=json.loads((O/'SOLAR_TRAIN_HISTORY.json').read_text())
 pq=json.loads((O/'SOLAR_REAL_LOCAL_PQ_REPORT.json').read_text())
 test=json.loads((O/'SOLAR_KAGGLE_TEST_AUDIT.json').read_text())
 assert test['images']==180
 best=pq['selection'];v=pq['date_disjoint_audit']
 c=canvas.Canvas(str(P),pagesize=letter,pageCompression=1)
 c.setTitle('Real GONG Solar Filament Segmentation - Technical Report')
 y=head(c,1,'1 | Objective and data')
 y=sub(c,'Objective',y)
 y=para(c,'Segment each dark, curvilinear solar chromospheric filament as a distinct instance in NSO GONG 2048 x 2048 H-alpha frames. The competition emphasizes Panoptic Quality (PQ), accurate shapes and reproducibility.',y)
 y=sub(c,'Official participant-provided MAGFiLO 2026 data',y)
 y=bul(c,['The two ZIPs are segments of one Kaggle training/test data upload.',
          '707 distinct annotated training images: 589 in 00.zip and 118 in MAGFiLO Kaggle archive.',
          '1,154 annotated image views, from multiple raters; 180 separate unlabeled competition test frames.',
          'All annotations grouped by image and observation DATE before splitting.'],y)
 y=sub(c,'Experimental partitions',y)
 y=para(c,'582 images for training; 63 tuning images for checkpoint and operating thresholds; 62 held-out date-separated images for diagnostic audit. Kaggle test labels were never accessed.',y)
 y=sub(c,'The technical contribution',y)
 y=bul(c,['CPU-trainable compact U-Net with positive-crop sampling and brightness augmentation.',
          'Multi-rater instance PQ, conservative object filters and COCO compressed RLE.',
          'Reproducible model artifact, data provenance, transparent errors and 180-image inference output.'],y)
 if y>340:
  c.drawImage(ImageReader(str(O/'SOLAR_REAL_TUNE_ROW_ERROR_PANEL.png')),65,y-230,width=480,height=185,preserveAspectRatio=True)
  y=para(c,'Real GONG images: green ground truth only; red prediction only; cyan overlap (512px diagnostic).',y-247,size=8.1,leading=11)
 c.showPage()
 y=head(c,2,'2 | Model and methodology')
 y=sub(c,'Training',y)
 y=para(c,'Source 2048 x 2048 JPEGs are area-downsampled to 512 x 512 for CPU training. The original polygon annotations are rasterized separately per annotator into label maps. Training takes stochastic 256 x 256 crops, with a bias toward positive pixels.',y)
 y=para(c,'UNet base channels 8, AdamW optimizer, learning rate 0.0006, weight decay 0.0001, positive-weighted binary cross entropy plus a soft Dice loss. The best checkpoint is selected by full-image tuning loss. No external pretrained backbone or Kaggle test labels are used.',y)
 y=sub(c,'Training result recorded',y)
 y=para(c,f"Completed {len(hist['history'])} epochs in the recovered execution. First tuning loss {hist['history'][0]['tune_loss']:.4f}; best tuning loss {hist['best_tune_loss']:.4f}. Checkpoint and every epoch's logs are archived, including loss and CPU time.",y)
 y=sub(c,'Instance postprocessing',y)
 y=bul(c,['Compute sigmoid probability for each 512-pixel solar image.',
          'Identify solar disk; discard limb artifacts and low-confidence pixels.',
          'Close tiny gaps, connected components, minimum area and elongation filtering.',
          'Upsample each sparse instance mask 4x and encode exact 2048-pixel COCO column-major RLE.',
          'Select threshold and minimum area from 63-image tuning partition only.'],y)
 y=sub(c,'Metric',y)
 y=para(c,'One-to-one matching of predicted and ground-truth instances when IoU > 0.5. Aggregate matched IoU, true positives, false positives and false negatives across distinct annotator views before computing PQ.',y)
 if y>290:
  c.drawImage(ImageReader(str(O/'SOLAR_10EPOCH_REAL_TRAINING_CURVE.png')),65,y-190,width=480,height=185,preserveAspectRatio=True)
 c.showPage()
 y=head(c,3,'3 | Quantitative results and limitations')
 y=sub(c,'Real-data local tuning score',y)
 y=para(c,f"Pooled PQ = {best['tune_pq']['pq']:.4f} across 63 tuning images. Threshold and object size settings: {best['config']}.",y)
 y=sub(c,'Date-separated real-data diagnostic audit',y)
 a=v['pooled_pq']
 y=para(c,f"Pooled PQ = {a['pq']:.4f} across {v['n_holdout']} date-separated images. TP = {a['tp']}; FP = {a['fp']}; FN = {a['fn']}. This is LOCAL 512-pixel PQ, not the organizer's 2048-pixel hidden-test score.",y)
 y=sub(c,'Full unlabeled Kaggle test inference',y)
 y=para(c,f"Processed {test['images']} original 2048-pixel images and wrote {test['predictions']} predicted filament instance rows. File checksum SHA-256: {test['sha256'][:30]}...; no test ground truth was read.",y)
 native_file=O/'SOLAR_NATIVE2048_ORIGINAL_POLYGON_AUDIT.json'
 if native_file.exists():
  native=json.loads(native_file.read_text())
  y=para(c,f"Original 2048px polygon audit: pooled local PQ {native['pooled_pq']['pq']:.4f} on {native['date_separated_subset']} dated-disjoint audit images. This reveals native boundary loss from 512px model inference.",y,size=9.1,leading=12)
 y=sub(c,'Failure modes and scientific threats',y)
 y=bul(c,['512-pixel training can destroy narrow filament strands and alter boundaries.',
          'Sunspot-like regions, limb structures and irregular dark patches can generate false detections.',
          'Multiple annotation views disagree about boundaries and how to split branching structures.',
          'The held-out diagnostic data are excluded from training and tuning, but not a secret official holdout.',
          'No official Kaggle leaderboard result or award is claimed; code and report need participant approval.'],y)
 if y>200:
  c.drawImage(ImageReader(str(O/'SOLAR_REAL_TUNE_ROW_ERROR_PANEL.png')),65,y-125,width=480,height=125,preserveAspectRatio=True)
 c.showPage()
 y=head(c,4,'4 | Reproduction and contest compliance')
 y=sub(c,'Reproduce from legitimate competition ZIP files',y)
 y=bul(c,['Keep input data in their original filenames. The project reads ZIP member paths directly.',
          'python solar_real.py: rebuild 512px labeled image cache and refit model.',
          'python solar_eval.py: locally select segmentation thresholds, run held-out PQ, build CSV.',
          'Check prediction filament_id uniqueness, COCO RLE shapes, tests and SHA256 manifest.',
          'For official Kaggle: participant must accept rules, publish required source/report and submit the candidate CSV.'],y)
 y=sub(c,'Artifact inventory',y)
 y=para(c,'Checkpoint .pt; source code; participant-upload data manifest; date split; train/tuning histories; pooled multi-rater PQ JSON; CSV candidate and audit; reproducer tests; this four-page technical report. The user-uploaded 700MB image archives are not redistributed in the package.',y)
 y=sub(c,'Attribution and AI disclosure',y)
 y=para(c,'Research software, analysis and technical report drafting were AI-assisted. The applicant is responsible for reviewing all code and statements, confirming the rules, meeting participation/award eligibility, and uploading the original source repository and submission.',y)
 y=sub(c,'Official contest',y)
 y=para(c,'https://www.kaggle.com/competitions/filament-segmentation-2026',y)
 y=sub(c,'Go / no-go statement',y)
 y=para(c,'A syntactically valid CSV is NOT sufficient evidence of award-level performance. The current model is a truthful competitive engineering candidate with measured local results; additional full-resolution validation and official private evaluation are required.',y)
 c.showPage();c.save()
 return P

if __name__=='__main__':print(generate())
