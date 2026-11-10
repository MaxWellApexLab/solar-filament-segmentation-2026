"""Rigorous instance/multirater PQ diagnostics and leakage-safe ablation.

Official PQ denominator calculated per independent annotator view then pooled.
No test labels are loaded by this module. A 1-to-many / many-to-1 IoU graph
is reported separately to uncover fragmentation and merging failure modes.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import pq_counts,pooled_pq


def failure_topology(gt,pred,overlap_cutoff=.1):
    gt=np.asarray(gt);pred=np.asarray(pred)
    if gt.shape!=pred.shape or gt.ndim!=2:raise ValueError('2D instance labels same shape')
    g=[v for v in np.unique(gt) if v>0];p=[v for v in np.unique(pred) if v>0]
    graph_g={int(i):[] for i in g};graph_p={int(i):[] for i in p}
    # full masks computed at 2d; O(n_instances x sample_size) but diagnostic use
    for i in g:
        gm=gt==i
        neighbors=np.unique(pred[gm])
        for j in neighbors:
            if j==0:continue
            pm=pred==j
            iou=np.count_nonzero(gm&pm)/np.count_nonzero(gm|pm)
            if iou>overlap_cutoff:
                graph_g[int(i)].append(int(j));graph_p[int(j)].append(int(i))
    return {'fragmented_GT_instances':sum(len(v)>1 for v in graph_g.values()),
            'merged_prediction_instances':sum(len(v)>1 for v in graph_p.values()),
            'matching_gt':len(g),'matching_pred':len(p),
            'note':'IoU-graph counts are diagnostics, NOT organizer-defined fragmentation penalty'}


def evaluate_annotators(annotations, prediction):
    """A list of independent human label maps for one observation, ONE prediction."""
    if len(annotations)==0:raise ValueError('must supply at least one human annotation view')
    counts=[pq_counts(gt,prediction) for gt in annotations]
    pooled=pooled_pq(counts)
    topology=[failure_topology(gt,prediction) for gt in annotations]
    return {'pq':pooled['pq'],'tp':pooled['tp'],'fp':pooled['fp'],
            'fn':pooled['fn'],'sum_iou':pooled['total_iou'],
            'fragmentation_diagnostics':topology,
            'annotators':len(annotations)}


def aggregate_observations(results):
    pooled=pooled_pq([{'tp':r['tp'],'fp':r['fp'],'fn':r['fn'],'sum_iou':r['sum_iou']} for r in results])
    return {'pooled_multi_annotator_pq':pooled['pq'], 'tp':pooled['tp'],
            'fp':pooled['fp'],'fn':pooled['fn'], 'images':len(results),
            'note':'local independent annotator pooling, not organizer/private score'}


def split_dates_strict(records,seed=2026,holdout_fraction=.2):
    """List (image_path, annotation_list), filename date YYYYMMDD at start.

    Date-disjoint, not observation-disjoint or ann-ID-disjoint: same day/observatory
    atmospheric conditions cannot appear in both folds.
    """
    from .dataset import date_holdout
    names={Path(name).name:None for name,_ in records}
    tr,va=date_holdout(names,ratio=holdout_fraction,seed=seed)
    a,b=set(tr),set(va)
    if a&b:raise AssertionError('data leak')
    if set(n[:8] for n in a)&set(n[:8] for n in b):raise AssertionError('date leak')
    return [r for r in records if Path(r[0]).name in a],[r for r in records if Path(r[0]).name in b]
