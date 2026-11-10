"""Full original 2048px polygon vs predicted-upscaled instance PQ audit.

NO hidden Kaggle test labels. A precommitted hash-ordered sample from the
participant-provided dated-disjoint 62-image local audit partition.
"""
import json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from solar_real import ROOT,RESULT,CACHE,groups,read_model,predict
from solar_eval import split,instance_map
from solar_filament.metrics import pq_counts,pooled_pq


def rasterize_original(anns):
    lab=np.zeros((2048,2048),np.int16);count=0
    for ann in anns:
        seg=ann.get('segmentation')
        if not isinstance(seg,list):continue
        m=np.zeros((2048,2048),np.uint8)
        for poly in seg:
            xy=np.asarray(poly,dtype=np.float32).reshape(-1,2)
            if len(xy)>=3:cv2.fillPoly(m,[np.rint(xy).astype(np.int32)],1)
        if m.any():
            count+=1;lab[(m>0)&(lab==0)]=count
    return lab


def audit(n=10):
    info=json.loads((RESULT/'SOLAR_REAL_LOCAL_PQ_REPORT.json').read_text())
    config=info['selection']['config'];net=read_model()
    rows=sorted(split()['holdout'],key=lambda r:hashlib.sha256(('solar-2048-sample-2026-'+r['name']).encode()).hexdigest())[:n]
    g=groups();tallies=[]
    for i,r in enumerate(rows):
        with np.load(CACHE/r['cache_file']) as d:gray=d['im']
        m=instance_map(predict(net,gray),gray,**config)
        predictions=np.repeat(np.repeat(m,4,axis=0),4,axis=1)
        for anns in g[r['name']]:
            gt=rasterize_original(anns)
            tallies.append(pq_counts(gt,predictions))
        print('ORIGINAL 2048px',i+1,'of',len(rows),'views',len(g[r['name']]),flush=True)
    report={'metric':'COCO polygon instance PQ at full native 2048 pixel resolution',
            'source':'participant-supplied MAGFiLO 2026 training annotations, not hidden test',
            'date_separated_subset':len(rows),'annotator_views':len(tallies),
            'pooled_pq':pooled_pq(tallies),
            'note':'Precommitted hash selection but data were previously used in earlier local audits; NOT a blind final organizer score; input images still downsampled 512 model.',
            'files':[x['name'] for x in rows]}
    (RESULT/'SOLAR_NATIVE2048_ORIGINAL_POLYGON_AUDIT.json').write_text(json.dumps(report,indent=2))
    print('FULL-RES',json.dumps(report['pooled_pq']),flush=True)
    return report

if __name__=='__main__':audit()
