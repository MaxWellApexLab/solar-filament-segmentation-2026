"""Real Solar 2026 MAGFiLO native 2048px edge-clipped TTA candidate.

Fits NOTHING on Kaggle test. Uses project checkpoint and original user-supplied
Kaggle test JPGs. The threshold 0.45 was chosen using annotated training data;
local validation must not be interpreted as hidden leaderboard PQ.
"""
import csv,hashlib,json,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from solar_real import ROOT,RESULT,Z2,read_model
from solar_eval import instance_map
from solar_tta import predict_tta
from solar_filament.coco_rle import counts_to_string


def edge_refine(inst,prob,threshold=.45):
    native=cv2.resize(inst,(2048,2048),interpolation=cv2.INTER_NEAREST)
    continuous=cv2.resize(prob,(2048,2048),interpolation=cv2.INTER_LINEAR)
    return np.where(continuous>=threshold,native,0).astype(np.int16)


def sparse_rle(labels,label):
    """Build global column-major COCO RLE from a sparse object bbox."""
    x,y,w,h=cv2.boundingRect(np.uint8(labels==label))
    if not w or not h:return None
    H,W=labels.shape;intervals=[]
    for col in range(x,x+w):
        yy=np.flatnonzero(labels[y:y+h,col]==label)
        if not len(yy):continue
        breaks=np.flatnonzero(np.diff(yy)>1)+1
        for run in np.split(yy,breaks):
            start=col*H+y+int(run[0]);intervals.append((start,start+len(run)))
    intervals.sort();counts=[];pos=0
    for start,end in intervals:
        if start==pos and len(counts)>=2:counts[-1]+=end-start
        else:counts.extend((start-pos,end-start))
        pos=end
    if pos<H*W:counts.append(H*W-pos)
    if sum(counts)!=H*W:raise ValueError('invalid COCO RLE')
    return counts_to_string(counts)


def generate(out):
    model=read_model()
    cfg=json.loads((RESULT/'SOLAR_TTA_COMPARISON.json').read_text())['parameters']
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    total=0;per=[]
    with zipfile.ZipFile(Z2) as z,out.open('w',newline='') as f:
        paths=sorted(n for n in z.namelist() if '/test/test_images/' in n and n.endswith('.jpeg'))
        if len(paths)!=180:raise ValueError('expected exactly 180 official test images')
        writer=csv.writer(f);writer.writerow(('filament_id','segmentation_rle'))
        for n in paths:
            img=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),cv2.IMREAD_GRAYSCALE)
            if img.shape!=(2048,2048):raise ValueError('bad input image')
            small=cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
            proba=predict_tta(model,small)
            labels=instance_map(proba,small,**cfg)
            final=edge_refine(labels,proba)
            k=0
            for old in range(1,int(labels.max())+1):
                coded=sparse_rle(final,old)
                if coded is None:continue
                k+=1;writer.writerow((Path(n).stem+'_'+str(k),coded))
            total+=k;per.append((Path(n).name,k))
    audit={'images':len(per),'filaments':total,'empty_images':sum(x[1]==0 for x in per),
           'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'official_score':None,
           'testing_labels_used':False,'edge_threshold':.45,'candidate':out.name}
    out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
    return audit
