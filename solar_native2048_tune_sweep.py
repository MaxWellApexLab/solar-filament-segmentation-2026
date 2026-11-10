"""High-resolution TTA segmentation parameter scan using TRAIN/TUNE split only.

The prior 62-image audit subset is never read. This scan is diagnostic, not a
new untouched heldout. A changed candidate requires an additional audit.
"""
from pathlib import Path
import json,hashlib,numpy as np,cv2
from solar_real import CACHE,RESULT,groups,read_model
from solar_eval import split,instance_map
from solar_tta import predict_tta
from solar_fullres_check import rasterize_original
from solar_filament.metrics import pq_counts,pooled_pq


def search(n=30):
    names=sorted(split()['tune'],key=lambda r:hashlib.sha256(('solar-native-tune-'+r['name']).encode()).hexdigest())[:n]
    views=groups();net=read_model()
    cfg=[{'threshold':threshold,'min_area':area,'min_elongation':1.2}
         for threshold in (.3,.4,.5) for area in (35,70,110)]
    scores=[[] for _ in cfg]
    for j,row in enumerate(names):
        with np.load(CACHE/row['cache_file']) as p:im=p['im']
        prob=predict_tta(net,im)
        gs=[rasterize_original(x) for x in views[row['name']]]
        for k,c in enumerate(cfg):
            l=instance_map(prob,im,**c)
            up=cv2.resize(l,(2048,2048),interpolation=cv2.INTER_NEAREST)
            for g in gs:scores[k].append(pq_counts(g,up))
        if (j+1)%5==0:print('NATIVE TUNE',j+1,'/',len(names),flush=True)
    metrics=[{'config':c,'native_tune_pq':pooled_pq(s)} for c,s in zip(cfg,scores)]
    metrics.sort(key=lambda x:-x['native_tune_pq']['pq'])
    reference={'threshold':.4,'min_area':70,'min_elongation':1.2}
    baseline=next(t for t in metrics if t['config']==reference)['native_tune_pq']['pq']
    out={'tuning_source':'30 hash-selected official annotated TRAIN/tune images',
         'annotator_views':len(scores[0]),'reference_512px_selected_config':reference,
         'reference_native_tune_pq':baseline,'results':metrics,'best_config':metrics[0]['config'],
         'best_native_tune_pq':metrics[0]['native_tune_pq']['pq'],
         'note':'This is a parameter ablation on tuning only; no guarantee of generalization or organizer score'}
    (RESULT/'SOLAR_NATIVE2048_TUNE_SWEEP.json').write_text(json.dumps(out,indent=2))
    print('TUNE_FINISHED',json.dumps({'baseline':baseline,'new_best':out['best_native_tune_pq'],'config':out['best_config']}),flush=True)
    return out

if __name__=='__main__':search()
