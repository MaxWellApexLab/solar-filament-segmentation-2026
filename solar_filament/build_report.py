"""Create four-page preliminary competition technical report (explicitly NOT final)."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'release'/'SOLAR_FILAMENT_4PAGE_TECHNICAL_DRAFT.pdf'
res=json.loads((ROOT/'release/solar_synthetic_control.json').read_text())
stress=json.loads((ROOT/'release/SOLAR_SYNTHETIC_HOLDOUT_BENCHMARK.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='SectionCustom',parent=styles['Heading2'],fontSize=13,spaceAfter=9,spaceBefore=15,textColor=HexColor('#143a57')))
styles.add(ParagraphStyle(name='BodyCustom',parent=styles['BodyText'],fontSize=10.5,leading=15.3,spaceAfter=8,textColor=HexColor('#23384b')))
styles.add(ParagraphStyle(name='SmallCustom',parent=styles['BodyText'],fontSize=8.7,leading=12,spaceAfter=5,textColor=HexColor('#385570')))
styles.add(ParagraphStyle(name='ReportTitle',parent=styles['Heading1'],fontSize=20,leading=25,textColor=HexColor('#102b46'),spaceAfter=14))

def P(txt,style='BodyCustom'):return Paragraph(txt,styles[style])

def section(head,text):return [P(head,'SectionCustom'),P(text)]

def foot(canvas,doc):
    canvas.saveState()
    w,h=letter;canvas.setStrokeColor(HexColor('#cdd9e2'));canvas.line(52,53,w-52,53)
    canvas.setFont('Helvetica',8);canvas.setFillColor(HexColor('#677f92'))
    canvas.drawString(52,39,'PrizeStrike • pre-submission engineering candidate • generated 2026-11-04')
    canvas.drawRightString(w-52,39,str(doc.page)+' / 4')
    canvas.restoreState()

pages=[]
# Page 1
pages += [P('Solar Filament Segmentation 2026','ReportTitle'),
P('<b>Technical report — PRELIMINARY. No organizer train or held-out data were available in this environment; no official leaderboard score is claimed.</b>'),
P('Research objective','SectionCustom'),
P('Segment each dark chromospheric filament as a distinct non-overlapping instance in NSO GONG H-alpha observations. Preserve fine structures while preventing fragmented detections and merged nearby filaments. The official assessment weights Panoptic Quality and related geometry (70%) plus scientific presentation, morphology and reproducibility (30%).'),
P('Input/output and restrictions','SectionCustom'),
P('Image input: 2048 × 2048 H-alpha JPG. Output: one CSV row per predicted filament, with <font name="Courier">filament_id</font> and <font name="Courier">segmentation_rle</font> columns. RLE masks are column-major COCO compressed counts. Train only with the organizer-provided MAGFiLO training annotations; avoid public releases that might reveal challenge test labels.'),
P('Visual sanity check — explicitly synthetic, not GONG data','SectionCustom'),
Table([[Image(str(ROOT/'figures/solar_SYNTHETIC_input.png'),width=218,height=218),Image(str(ROOT/'figures/solar_SYNTHETIC_prediction.png'),width=218,height=218)]],colWidths=[220,220]),
P('Left: artificial solar disc with three injected lines and a sunspot-like distractor. Right: candidate instance masks. This illustration is a software regression test only.','SmallCustom'),
PageBreak()]
# Page 2
pages += [P('Method: contrast-aware segmentation','ReportTitle')]
for head,text in [
('Stage 1 — robust solar-disc localization','Estimate the dominant solar disk from blurred intensity, then erode the limb boundary to exclude background and high-gradient off-disk structures. Error checking rejects crops and missing discs.'),
('Stage 2 — multi-scale dark-structure response','High-pass flatten large-scale intensity variation and compute elliptical black top-hat morphology at 9, 19 and 35 pixels. Combine the responses by maximum. Threshold connected dark curvilinear structures and close small gaps without broadly merging neighbours.'),
('Stage 3 — instance candidates and false-positive control','Connected components become provisional instances. Area and covariance eigenvalue aspect ratio reject tiny, round sunspot-like candidates. Input to these rules is image intensity alone; no external labels or test metadata.'),
('Stage 4 — optional learned UNet','A compact PyTorch U-Net (trainable from scratch) uses BCE + soft Dice on the organizer COCO annotations. For each physical image that multiple annotators label, sample a rater view per example rather than creating a fabricated consensus mask. Crop sampling favours positive filaments; brightness augmentation addresses variable observing conditions.'),
('Leakage control','Split by physical observation date. File IDs that refer to the same underlying GONG image must remain in the same training or validation group. Compare learned model to morphology-only baseline, and pool TP/FP/FN across all distinct annotator views before PQ aggregation.')]:
    pages += section(head,text)
pages += [P('Our experimental code path can be reproduced with no pretrained weights. No U-Net weights are included because official training images were not accessible during this run.','SmallCustom'),PageBreak()]
# Page 3
pages += [P('Verification and current results','ReportTitle'),
P('Panoptic Quality','SectionCustom'),
P('Predicted and annotated objects are matched one-to-one when Intersection-over-Union exceeds 0.5. PQ equals the sum of matched IoUs divided by TP + 0.5 FP + 0.5 FN. The official challenge scores multi-annotator image views separately before pooling.'),
P('Blind-style holdout on synthetic generated images','SectionCustom')]
rows=[['Pipeline','GT objects','Predicted objects','PQ','TP / FP / FN'],
['Legacy morphology', '28', '66', f"{stress['legacy_holdout']['pooled']['pq']:.3f}",
 f"{stress['legacy_holdout']['pooled']['tp']} / {stress['legacy_holdout']['pooled']['fp']} / {stress['legacy_holdout']['pooled']['fn']}"],
['Structure-aware trial', '28', '59', f"{stress['structure_holdout']['pooled']['pq']:.3f}",
 f"{stress['structure_holdout']['pooled']['tp']} / {stress['structure_holdout']['pooled']['fp']} / {stress['structure_holdout']['pooled']['fn']}"]]
t=Table(rows,colWidths=[143,69,99,53,100],rowHeights=31)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e6f1f6')),('GRID',(0,0),(-1,-1),.5,HexColor('#b8ced9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('FONTSIZE',(0,0),(-1,-1),8.4)]))
pages += [t,Spacer(1,10),P('<b>Result:</b> New structure-aware morphology was tuned on SIX separate synthetic scenes, but underperformed on SEVEN untouched synthetic scenes. It is retained for ablation, NOT promoted as the final algorithm. Neither PQ value was measured on GONG or MAGFiLO images. We do not tune with official hidden test labels.'),
P('Engineering checks','SectionCustom'),
P('Tests now cover 2048-compatible tiled U-Net inference, compressed column-major RLE, pooled multi-annotator PQ, date-separated training, malformed-data refusal, exact orbital numerical checks, climate-diagnostic and grid-memory rules. Local unit tests do NOT substitute organizer score.'),
P('Threats to validity','SectionCustom'),
P('GONG image quality and limb darkness vary by site and seeing. Real filaments include branching, overlapping, small structures, partial occlusion and multiple annotators with material disagreement. Morphological filters alone may systematically miss short or irregular filaments. A tiled U-Net inference pipeline is implemented; no official model weights or validation are available. Detector fusion requires an external labeled assessment before claiming improved PQ.'),
PageBreak()]
# Page 4
pages += [P('Reproducibility and finalization gates','ReportTitle'),
P('Code and reproducibility','SectionCustom'),
P('Run <font name="Courier">PYTHONPATH=. pytest -q tests</font> from the root. Solar predictions: <font name="Courier">python -m solar_filament.segment --images TEST_JPG... --output submission.csv</font>. Train: <font name="Courier">python -m solar_filament.train --image-root TRAIN_IMAGES --annotations MAGFiLO_TRAIN.json --output weights.pt</font>. The tiled U-Net inference enforces 2048-pixel official inputs; a separate preflight rejects non-organizer training annotations, and the CSV writer uses COCO compressed RLE.'),
P('Required official-data evaluation before final submission','SectionCustom'),
P('1. Accept Kaggle competition terms with a personally eligible account and obtain the 707 official training observations, corresponding COCO annotations and 180 test observations. Do not retrieve test labels from another public MAGFiLO source.\n2. Record dataset file hashes and observing-date split IDs.\n3. Optimize morphology and U-Net variants on the held-out official training dates, report pooled multi-rater PQ, IoU, Dice, model efficiency and false detections.\n4. Inspect filament morphology; include the editable Jupyter notebook; validate the predictions on Kaggle. Save submitted artifact ID and external score.\n5. Replace this preliminary PDF with experimental results, public Git repository URL, and reproducibility metadata.'),
P('External access and data rights','SectionCustom'),
P('Official National Solar Observatory GONG H-alpha image collection: https://nso.edu/data/nisp-data/h-alpha/. Official annotated challenge and submission rules: https://www.kaggle.com/competitions/filament-segmentation-2026/. Original COCO compressed RLE reference implementation: https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c.'),
P('Disclosure / attribution','SectionCustom'),
P('The draft was generated with assistance from OpenAI GPT-6, and the software was developed with AI assistance. No independent researchers, clinical experts or organizers are claimed as partners. The author must confirm all contributions, team identity, public licensing, and award eligibility before submission.'),
P('Finalization status','SectionCustom'),
P('<b>BLOCKED:</b> official labeled training set and organizer leaderboard evaluation have not been performed. This is therefore NOT a completed, award-competitive submission package. Only the reusable engineering baseline and truthful report draft are delivered.'),]

def main():
    doc=SimpleDocTemplate(str(OUT),pagesize=letter,leftMargin=52,rightMargin=52,topMargin=52,bottomMargin=65,title='Solar Filament Segmentation - Preliminary Report')
    doc.build(pages,onFirstPage=foot,onLaterPages=foot)
    print(OUT)
if __name__=='__main__':main()
