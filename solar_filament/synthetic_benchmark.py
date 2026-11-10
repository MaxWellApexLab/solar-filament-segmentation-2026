"""Blind-style *synthetic engineering stress test* for solar segmenters.

This is not official MAGFiLO performance and is not used to train on withheld
human annotations. Test textures explicitly synthesized by computer code.
Reports named baselines without deciding results in advance.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .segment import segment as legacy
from .structure_aware import segment as structure,Parameters
from .metrics import pooled_pq,pq_counts
from .science_eval import failure_topology


def generate(seed,size=384,n_filaments=4,noise_sigma=6.,limb_strength=.35):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[0:size,0:size]
    radius=size*.43;dist=np.sqrt((xx-size/2)**2+(yy-size/2)**2)
    inside=dist<radius
    disk=167*(1-limb_strength*(dist/radius)**2)
    noise=rng.normal(0,noise_sigma,(size,size))
    im=np.clip(disk+noise,0,255).astype('uint8')
    im[~inside]=np.uint8(np.clip(rng.normal(8,2,np.count_nonzero(~inside)),0,255))
    # Smooth dark plages (background texture) not all annotated filaments.
    for _ in range(5):
        px=int(rng.uniform(size*.25,size*.75));py=int(rng.uniform(size*.25,size*.75))
        if (px-size/2)**2+(py-size/2)**2>(radius-20)**2:continue
        cv2.circle(im,(px,py),int(rng.uniform(7,17)),int(rng.uniform(40,100)),-1)
    truth=np.zeros_like(im,dtype=np.int32)
    for j in range(1,n_filaments+1):
        accepted=False
        for _ in range(50):
            center=(int(rng.uniform(size*.25,size*.75)),int(rng.uniform(size*.25,size*.75)))
            length=rng.uniform(size*.12,size*.25)
            angle=rng.uniform(0,np.pi)
            u=np.array([np.cos(angle),np.sin(angle)]);n=np.array([-u[1],u[0]])
            points=[]
            for t in np.linspace(-1,1,14):
                coord=np.array(center)+.5*length*t*u+5*np.sin(3*t+seed)*n
                points.append(np.round(coord).astype(int))
            path=np.asarray(points,dtype='int32')
            if np.all((path[:,0]-size/2)**2+(path[:,1]-size/2)**2<(radius-17)**2):
                accepted=True;break
        if not accepted:continue
        thick=int(rng.integers(2,5));label=np.zeros_like(im)
        cv2.polylines(label,[path.reshape(-1,1,2)],False,j,thick)
        # Strong annotations are allowed at overlapping pixels from earlier
        # filaments only if no other label; don't create physically merged GT.
        region=(label>0)&(truth==0)
        truth[region]=j
        im[region]=np.uint8(rng.uniform(13,35))
    return im,truth


def score_suite(seeds,method,kwargs):
    measures=[];details=[]
    for seed in seeds:
        image,truth=generate(seed,noise_sigma=(3.+seed%6),limb_strength=.22+.04*(seed%5))
        pred,info=method(image,**kwargs)
        counts=pq_counts(truth,pred)
        measures.append(counts)
        details.append({'seed':seed,'n_predicted':int(pred.max()),'n_reference':int(truth.max()),
                        'tp':counts['tp'],'fp':counts['fp'],'fn':counts['fn'],
                        'pq':float(pooled_pq([counts])['pq']),
                        'topology':failure_topology(truth,pred)})
    return {'pooled':pooled_pq(measures),'individual':details}


def benchmark(output,save_example=False):
    # For transparent tuning: select parameters ONLY by synthetic train seeds.
    train=list(range(20,26));validation=list(range(70,77))
    assert not set(train)&set(validation)
    basetrain=score_suite(train,legacy,{'scales':(7,19,35),'threshold':21,'min_area':25,'min_elongation':1.5})
    bareval=score_suite(validation,legacy,{'scales':(7,19,35),'threshold':21,'min_area':25,'min_elongation':1.5})
    candidates=[]
    for percentile in [76,84,92]:
        for minarea in [18,30]:
            params=Parameters(response_percentile=percentile,line_min_area=minarea,min_elongation=1.4)
            train_score=score_suite(train,structure,{'params':params})['pooled']['pq']
            candidates.append((train_score,params))
    _,selected=max(candidates,key=lambda item:item[0])
    newval=score_suite(validation,structure,{'params':selected})
    data={'dataset':'SYNTHETIC ONLY / artificially rendered solar disks and filaments',
          'seed_train':train,'seed_holdout':validation,
          'validation_not_used_for_tuning':True,
          'legacy_train_pq':basetrain['pooled']['pq'],
          'legacy_holdout':bareval,
          'structure_holdout':newval,
          'structure_selected_on_train':selected.__dict__,
          'winner_on_holdout':'structure' if newval['pooled']['pq']>bareval['pooled']['pq'] else 'legacy',
          'source':'simulation; no official MAGFiLO labels or leaderboard'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2))
    if save_example:
        im,truth=generate(validation[0]);pred,_=structure(im,selected)
        side=np.concatenate([im, np.uint8(np.clip(truth*40,0,255)),np.uint8(np.clip(pred*40,0,255))],axis=1)
        Image.fromarray(side).save(output.with_suffix('.png'))
    return data

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    args=p.parse_args();r=benchmark(args.out,True)
    print(json.dumps({'legacy_holdout_pq':r['legacy_holdout']['pooled']['pq'],
                     'structure_holdout_pq':r['structure_holdout']['pooled']['pq'],
                     'selection':r['structure_selected_on_train']},indent=2))
