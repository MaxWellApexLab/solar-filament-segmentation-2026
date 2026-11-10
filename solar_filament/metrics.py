"""Instance Panoptic Quality (PQ) with 1:1 IoU>0.5 matching.

Our metric implements the standard PQ object-pair matching. The official scoring
has multi-rater pooling; call this metric separately for every annotated view
and aggregate the TP/FP/FN and total IoU across views, never silently average PQ.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def pq_counts(gt,pred,iou_threshold=.5):
    gt=np.asarray(gt);pred=np.asarray(pred)
    if gt.shape!=pred.shape or gt.ndim!=2:raise ValueError('integer instance masks must have identical 2D shape')
    ga=[int(x) for x in np.unique(gt) if x!=0]
    pa=[int(x) for x in np.unique(pred) if x!=0]
    if not ga and not pa:return {'tp':0,'fp':0,'fn':0,'sum_iou':0.0}
    if not ga:return {'tp':0,'fp':len(pa),'fn':0,'sum_iou':0.0}
    if not pa:return {'tp':0,'fp':0,'fn':len(ga),'sum_iou':0.0}
    # Local ROI match matrix, no pixelwise loops over labels and pairs.
    area_g={x:int(np.count_nonzero(gt==x)) for x in ga}
    area_p={x:int(np.count_nonzero(pred==x)) for x in pa}
    overlap=np.zeros((len(ga),len(pa)),float)
    gindex={v:i for i,v in enumerate(ga)};pindex={v:i for i,v in enumerate(pa)}
    both=(gt!=0)&(pred!=0)
    if np.any(both):
        pairs,counts=np.unique(np.stack((gt[both],pred[both]),axis=1),axis=0,return_counts=True)
        for (g,p),intersection in zip(pairs,counts):
            i,j=gindex[int(g)],pindex[int(p)]
            overlap[i,j]=intersection/(area_g[int(g)]+area_p[int(p)]-intersection)
    ri,ci=linear_sum_assignment(-overlap)
    valid=overlap[ri,ci]>iou_threshold
    n=int(valid.sum());total=float(overlap[ri[valid],ci[valid]].sum())
    return {'tp':n,'fp':len(pa)-n,'fn':len(ga)-n,'sum_iou':total}


def pooled_pq(counts):
    tp=sum(x['tp'] for x in counts);fp=sum(x['fp'] for x in counts);fn=sum(x['fn'] for x in counts)
    total=sum(x['sum_iou'] for x in counts)
    denom=tp+0.5*(fp+fn)
    return {'pq':float(total/denom) if denom else 1.0,'tp':tp,'fp':fp,'fn':fn,'total_iou':total,
            'note':'local PQ; official MAGFiLO evaluation pools over annotator views'}
