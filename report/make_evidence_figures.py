"""Evidence figures from saved real-data measurements and genuine organizer images.

No predictions on hidden test labels. The visualization uses unlabeled test JPEGs.
"""
from pathlib import Path
import csv,zipfile,json
import numpy as np,cv2,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solar_real import ROOT,RESULT,read_model,Z2
from solar_tta import predict_tta
from solar_eval import instance_map
from refined_inference import edge_refine
from solar_source import OfficialTestImages

OUT=ROOT/'report'/'figures';OUT.mkdir(parents=True,exist_ok=True)


def create():
    history=json.loads((RESULT/'SOLAR_TRAIN_HISTORY.json').read_text())['history']
    fig,ax=plt.subplots(figsize=(6.4,2.8))
    ax.plot([v['epoch'] for v in history],[v['training_loss'] for v in history],'-o',label='Training')
    ax.plot([v['epoch'] for v in history],[v['tune_loss'] for v in history],'-s',label='Date-isolated tuning')
    ax.set_xlabel('Epoch');ax.set_ylabel('BCE + soft Dice objective');ax.set_xticks(range(1,11))
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'training_curve.pdf');plt.close(fig)

    metric=json.loads((RESULT/'SOLAR_ORIGINAL2048_EDGE_REPRODUCED.json').read_text())
    methods=['Unaugmented\nU-Net','4-flip TTA','TTA + native\nedge clip']
    scores=[0.239523,metric['baseline_TTA']['pq'],metric['edge_refined_TTA']['pq']]
    fig,ax=plt.subplots(figsize=(5.4,2.6))
    ax.bar(methods,scores)
    ax.set_ylim(0,.35);ax.set_ylabel('Pooled native 2048px PQ')
    for i,v in enumerate(scores):ax.text(i,v+.006,f'{v:.4f}',ha='center',fontsize=8)
    ax.set_title('Real 62-image local audit (not private Kaggle test)',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'pq_ablation.pdf');plt.close(fig)

    # Real unlabeled test imagery used ONLY for qualitative illustration.
    counts={}
    with (RESULT/'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv').open() as fp:
        for row in csv.DictReader(fp):
            stem=row['filament_id'].rsplit('_',1)[0]
            counts[stem]=counts.get(stem,0)+1
    selected=[k for k,_ in sorted(counts.items(),key=lambda v:(-v[1],v[0]))[:1]]
    net=read_model()
    cfg=json.loads((RESULT/'SOLAR_TTA_COMPARISON.json').read_text())['parameters']
    with OfficialTestImages(Z2) as z:
        data=z.bytes(selected[0])
    original=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    image=cv2.resize(original,(512,512),interpolation=cv2.INTER_AREA)
    prob=predict_tta(net,image);lab=instance_map(prob,image,**cfg)
    edge=edge_refine(lab,prob,.45)
    edge512=cv2.resize((edge>0).astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
    fig,ax=plt.subplots(1,3,figsize=(11,3.3))
    for a,im,ttl in zip(ax,(image,prob,edge512),('Original H-alpha (downsampled)','Four-flip U-Net confidence','2048px edge-clipped mask (preview)')):
        a.imshow(im,cmap='gray' if im is image else 'magma',vmin=0 if im is prob else None,vmax=1 if im is prob else None)
        a.set_title(ttl,fontsize=9);a.set_axis_off()
    fig.suptitle(f'Official unlabeled test image {selected[0]}: morphology preview, no test GT',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'solar_test_prediction.pdf');plt.close(fig)
    print('FINISHED',sorted(x.name for x in OUT.iterdir()))

if __name__=='__main__':create()
