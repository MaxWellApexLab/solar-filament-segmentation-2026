"""Full-resolution morphological boundary calibration on original labeled solar data.

Calibration uses selected training TUNE dates only, never Kaggle test labels.
A separate dated-disjoint HOLDOUT subset is assessed after parameter selection.
It does not create a new learned segmentation network; controlled boundary test.
"""
import json,hashlib
import cv2,numpy as np
from solar_real import RESULT,CACHE,groups,read_model,predict
from solar_eval import split,instance_map
from solar_fullres_check import rasterize_original
from solar_filament.metrics import pq_counts,pooled_pq


def enlarge(labels,diameter):
    if diameter==0:return labels
    arr=np.asarray(labels,np.int16)
    out=arr.copy()
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*diameter+1,2*diameter+1))
    for k in range(1,int(arr.max())+1):
        grown=cv2.dilate(np.uint8(arr==k),kernel)>0
        out[(out==0)&grown]=k
    return out


def evaluate_group(net,rows,config,g,diameters=(0,2,4)):
    tally={d:[] for d in diameters}
    for j,r in enumerate(rows):
        with np.load(CACHE/r['cache_file']) as d:gray=d['im']
        labels=instance_map(predict(net,gray),gray,**config)
        up=np.repeat(np.repeat(labels,4,axis=0),4,axis=1)
        real=[rasterize_original(anns) for anns in g[r['name']]]
        for diameter in diameters:
            pred=enlarge(up,diameter)
            for gt in real:tally[diameter].append(pq_counts(gt,pred))
        print('FULL-RES CALIBRATION',j+1,'/',len(rows),flush=True)
    return {str(d):pooled_pq(tally[d]) for d in diameters}


def execute(n_tune=8,n_holdout=8):
    net=read_model();g=groups();info=json.loads((RESULT/'SOLAR_REAL_LOCAL_PQ_REPORT.json').read_text())
    config=info['selection']['config'];parts=split()
    s=lambda rows,key,n:sorted(rows,key=lambda r:hashlib.sha256((key+r['name']).encode()).hexdigest())[:n]
    tun=s(parts['tune'],'solar-2048-tune-'+str(2026),n_tune)
    hold=s(parts['holdout'],'solar-2048-holdout-'+str(2026),n_holdout)
    scores=evaluate_group(net,tun,config,g)
    winner=max(scores,key=lambda k:scores[k]['pq'])
    report={'calibration_source':'official participant provided MAGFiLO TRAIN polygon 2048px; 8 TUNE images, native 2048 coords',
       'method':'morphological dilation of 512px U-Net upscaled predictions',
       'tune_pq_by_dilation_px':scores,'chosen_dilation_px':int(winner),
       'holdout_pq':evaluate_group(net,hold,config,g,diameters=(int(winner),)),
       'not_a_blind_organizer_score':True,
       'data_leakage_boundary':'calibration uses date-grouped tuning dates, separate from holdout and unrelated to official test image labels'}
    (RESULT/'SOLAR_NATIVE2048_BOUNDARY_REFINEMENT.json').write_text(json.dumps(report,indent=2))
    print('REFINEMENT',json.dumps(report),flush=True)
    return report

if __name__=='__main__':execute()
