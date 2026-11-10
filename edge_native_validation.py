"""Reconstruct *original polygon* PQ for saved v5 edge-refined 180-test pipeline.

The official hidden-test labels are NEVER read. This is on 62 previously
examined TRAIN holdout observation dates and must NOT be called blind or an
official Kaggle score. One JSON record per original image enables safe resume.
Compare historical TTA metrics stored in v5 to identify drift before reporting
edge-refinement change. The 2048x2048 reference polygons come from authorized
MAGFiLO 2026 organizer TRAIN annotation JSON in the participant ZIP.
"""
from __future__ import annotations
import argparse,hashlib,json,os,zipfile
from pathlib import Path
import cv2,numpy as np
from solar_real import ROOT,RESULT,Z1,Z2,groups,read_model
from solar_tta import predict_tta
from solar_eval import instance_map
from solar_fullres_check import rasterize_original
from refined_inference import edge_refine
from solar_filament.metrics import pq_counts,pooled_pq

DATA=RESULT/'SOLAR_NATIVE2048_TTA_PAIRED.json'
CONFIG=RESULT/'SOLAR_TTA_COMPARISON.json'


def image_map(archive):
    return {Path(n).name:n for n in archive.namelist() if n.endswith('.jpeg') and ('/train/train_images/' in n or '/00/' in n)}


def _hash_json(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def compute_image(image_record,config,labels,model,za,zb,ma,mb,out_dir):
    name=image_record['image']
    out=Path(out_dir)/(Path(name).stem+'.json')
    # Cached records are validated against previous TTA counts, official
    # image name, configuration and checkpoint SHA at final assembly.
    if out.exists():return json.loads(out.read_text())
    archive,path=(za,ma[name]) if name in ma else (zb,mb[name])
    gray=cv2.imdecode(np.frombuffer(archive.read(path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape!=(2048,2048):raise ValueError('Bad annotated train image '+name)
    small=cv2.resize(gray,(512,512),interpolation=cv2.INTER_AREA)
    prob=predict_tta(model,small)
    inst=instance_map(prob,small,**config)
    original=cv2.resize(inst,(2048,2048),interpolation=cv2.INTER_NEAREST)
    refined=edge_refine(inst,prob,threshold=.45)
    refviews=labels[name]
    if len(refviews)!=len(image_record['counts']['tta']):raise ValueError('Annotation view mismatch '+name)
    reference_counts=[]; refined_counts=[]
    for i,ann in enumerate(refviews):
        gt=rasterize_original(ann)
        a=pq_counts(gt,original)
        saved=image_record['counts']['tta'][i]
        if (a['tp'],a['fp'],a['fn'])!=(saved['tp'],saved['fp'],saved['fn']) or abs(a['sum_iou']-saved['sum_iou'])>1e-6:
            raise ValueError(f'Saved TTA ground truth/count mismatch {name} annotator {i}: {a} != {saved}')
        reference_counts.append(a)
        refined_counts.append(pq_counts(gt,refined))
    record={'name':name,'date':image_record['date'],'annotator_views':len(refviews),
            'baseline_tta_counts':reference_counts,'edge_counts':refined_counts,
            'reference_matches_v5':True,'config':config,'edge_threshold':.45,
            'source_zip_filename':archive.filename,'zip_member':path,'jpeg_crc32':archive.getinfo(path).CRC}
    tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,out)
    return record


def run_batch(out_dir,first=0,count=10):
    base=json.loads(DATA.read_text())
    cfg=json.loads(CONFIG.read_text())['parameters']
    rows=base['image_details'][first:first+count]
    if not rows:raise ValueError('empty slice')
    all_labels=groups();model=read_model();Path(out_dir).mkdir(exist_ok=True,parents=True)
    results=[]
    with zipfile.ZipFile(Z1) as a,zipfile.ZipFile(Z2) as b:
        ma={Path(n).name:n for n in a.namelist() if n.endswith('.jpeg')}
        mb={Path(n).name:n for n in b.namelist() if '/train/train_images/' in n and n.endswith('.jpeg')}
        for i,row in enumerate(rows):
            r=compute_image(row,cfg,all_labels,model,a,b,ma,mb,out_dir)
            results.append(r)
            print('2048 GT edge',first+i+1,'/',base['images'],'rater views',r['annotator_views'],flush=True)
    return {'slice':[first,first+len(rows)],'count':len(rows),'computed_or_reused':len(results)}


def assemble(out_dir,out_path):
    base=json.loads(DATA.read_text());cfg=json.loads(CONFIG.read_text())['parameters']
    records=[]
    with zipfile.ZipFile(Z1) as a,zipfile.ZipFile(Z2) as b:
        ma={Path(n).name:n for n in a.namelist() if n.endswith('.jpeg')}
        mb={Path(n).name:n for n in b.namelist() if '/train/train_images/' in n and n.endswith('.jpeg')}
        for row in base['image_details']:
            name=row['image'];p=Path(out_dir)/(Path(name).stem+'.json')
            if not p.exists():raise FileNotFoundError('Missing real original-polygon evaluation '+name)
            rec=json.loads(p.read_text()); archive,path=(a,ma[name]) if name in ma else (b,mb[name])
            if rec['name']!=name or rec['config']!=cfg or rec['zip_member']!=path or rec['jpeg_crc32']!=archive.getinfo(path).CRC:
                raise ValueError('Stale or mismatched original image evidence '+name)
            if rec['annotator_views']!=len(row['counts']['tta']) or rec['baseline_tta_counts']!=row['counts']['tta']:
                raise ValueError('Drift from saved TTA reference '+name)
            records.append(rec)
    prev=pooled_pq([a for rec in records for a in rec['baseline_tta_counts']])
    after=pooled_pq([a for rec in records for a in rec['edge_counts']])
    if abs(prev['pq']-base['four_flip_tta']['pq'])>1e-8:raise ValueError('Stored v5 TTA PQ mismatch')
    rng=np.random.default_rng(20261105);n=len(records);deltas=[]
    for _ in range(2000):
        selected=rng.integers(0,n,size=n)
        b=pooled_pq([c for j in selected for c in records[j]['baseline_tta_counts']])
        t=pooled_pq([c for j in selected for c in records[j]['edge_counts']])
        deltas.append(t['pq']-b['pq'])
    result={'origin':'real original 2048x2048 organizer TRAIN polygon labels from user-provided MAGFiLO 2026 ZIPs',
            'n_images':len(records),'n_rater_views':sum(x['annotator_views'] for x in records),
            'baseline_TTA':prev,'edge_refined_TTA':after,'delta_pq':after['pq']-prev['pq'],
            'bootstrap_95_percentile_CI':list(map(float,np.percentile(deltas,[2.5,97.5]))),
            'independent_recomputed_TTA_matches_v5':True,
            'limitations':['v5 holdout previously inspected; NOT prospectively hidden test',
                           'image-level bootstrap assumes imperfect independence',
                           'pipeline predicts at 512px then scales, no native-detail learning',
                           'this is not a Kaggle official test leaderboard result'],
            'per_image_original_polygon_counts':records}
    dest=Path(out_path);dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(result,indent=2))
    print('ORIGINAL_POLYGON_2048_EDGE_AUDIT',json.dumps({'n':len(records),'TTA_pq':prev['pq'],'edge_pq':after['pq'],'delta':result['delta_pq'],'ci95':result['bootstrap_95_percentile_CI']}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['batch','assemble']);parser.add_argument('--evidence-dir',required=True)
    parser.add_argument('--first',type=int,default=0);parser.add_argument('--count',type=int,default=10);parser.add_argument('--output')
    a=parser.parse_args()
    r=run_batch(a.evidence_dir,a.first,a.count) if a.command=='batch' else assemble(a.evidence_dir,a.output or RESULT/'SOLAR_ORIGINAL2048_EDGE_REPRODUCED.json')
    print(json.dumps({k:v for k,v in r.items() if k!='per_image_original_polygon_counts'},indent=2))
