"""Paired native 2048px local audit of baseline and four-flip TTA.

Uses ONLY date-separated already-opened MAGFiLO train annotation split; this is
NOT a blind test and does not certify the official scoring pipeline. Per-image
bootstrap resamples dates/images, never individual annotators independently.
"""
import json, numpy as np
import cv2
from solar_real import CACHE, RESULT, groups,read_model,predict
from solar_eval import split,instance_map
from solar_tta import predict_tta
from solar_fullres_check import rasterize_original
from solar_filament.metrics import pq_counts,pooled_pq


def paired(n=None):
    cfg=json.loads((RESULT/'SOLAR_TTA_COMPARISON.json').read_text())['parameters']
    rows=split()['holdout'];rows=rows if n is None else rows[:n]
    anns=groups();net=read_model()
    individual=[]
    for j,row in enumerate(rows):
        with np.load(CACHE/row['cache_file']) as d: img=d['im']
        a=instance_map(predict(net,img),img,**cfg)
        b=instance_map(predict_tta(net,img),img,**cfg)
        # Nearest-neighbor mapping matches the exact Kaggle RLE upsampling.
        pred_a=cv2.resize(a,(2048,2048),interpolation=cv2.INTER_NEAREST)
        pred_b=cv2.resize(b,(2048,2048),interpolation=cv2.INTER_NEAREST)
        scores={'baseline':[],'tta':[]}
        for view in anns[row['name']]:
            gt=rasterize_original(view)
            scores['baseline'].append(pq_counts(gt,pred_a))
            scores['tta'].append(pq_counts(gt,pred_b))
        individual.append({'image':row['name'],'date':row['date'],'views':len(anns[row['name']]),'counts':scores})
        if (j+1)%10==0 or j==len(rows)-1:
            print(f'2048 full-res native polygon PQ {j+1}/{len(rows)}',flush=True)
    ref=pooled_pq([v for item in individual for v in item['counts']['baseline']])
    tta=pooled_pq([v for item in individual for v in item['counts']['tta']])
    rng=np.random.default_rng(20261104)
    diffs=[]
    for iteration in range(2000):
        sample=rng.integers(0,len(individual),len(individual))
        cb=[v for i in sample for v in individual[i]['counts']['baseline']]
        ct=[v for i in sample for v in individual[i]['counts']['tta']]
        diffs.append(pooled_pq(ct)['pq']-pooled_pq(cb)['pq'])
    low,hi=np.quantile(diffs,[.025,.975]);diff=float(tta['pq']-ref['pq'])
    report={'data':'participant-uploaded real MAGFiLO 2026 train annotations, pixel-native 2048x2048',
        'reference_comparison':'same 62 date-disjoint holdout images, same base U-Net, identical segmentation parameters',
        'images':len(rows),'annotator_views':sum(item['views'] for item in individual),
        'baseline':ref,'four_flip_tta':tta,'tta_minus_baseline_pq':diff,
        'paired_bootstrap_95_percentile_ci':[float(low),float(hi)],
        'bootstrap_unit':'whole image + all its annotators; 2000 resamples',
        'cautions':['Holdout images were opened in previous engineering iterations; not an untouched final blind test',
                    'Local PQ metric based on instance Hungarian IoU > 0.5, not organizer score',
                    'Predicted masks from 512px-trained model, nearest-neighbor scaled to 2048px',
                    'Bootstrap assumes image-level approximate independence despite time correlation'],
        'image_details':individual}
    (RESULT/'SOLAR_NATIVE2048_TTA_PAIRED.json').write_text(json.dumps(report,indent=2))
    print('2048_PAIRED_SUMMARY',json.dumps({k:report[k] for k in ('images','annotator_views','tta_minus_baseline_pq','paired_bootstrap_95_percentile_ci')})+' '+str(ref['pq'])+' to '+str(tta['pq']),flush=True)
    return report

if __name__=='__main__':paired()
