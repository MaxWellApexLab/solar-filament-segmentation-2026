"""Real Kaggle MAGFiLO 2026 instance-level PQ and unlabeled test CSV.

The 512px GT and predictions create local evaluation only. The heldout set
is observation-date-disjoint; no hidden test labels are accessed.
"""
from pathlib import Path
import csv,hashlib,json,math,random,zipfile
import cv2,numpy as np,torch
from solar_real import ROOT,RESULT,CACHE,Z2,groups,read_model,predict
from solar_filament.metrics import pq_counts,pooled_pq
from solar_filament.coco_rle import encode,decode,counts_to_string,string_to_counts


def instance_map(prob,gray,threshold=.6,min_area=25,min_elongation=1.3):
    blur=cv2.GaussianBlur(gray,(0,0),2)
    disk=(blur>max(10,float(np.percentile(blur,98))*.12)).astype(np.uint8)
    n,l,stats,cents=cv2.connectedComponentsWithStats(disk,8)
    if n>1:
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));disk=np.uint8(l==largest)
    disk=cv2.erode(disk,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    binary=np.uint8((prob>threshold)&(disk>0))
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    n,cc,stats,_=cv2.connectedComponentsWithStats(binary,8)
    lab=np.zeros_like(cc,np.int16);count=0
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i]);
        if area<min_area:continue
        roimask=cc[y:y+h,x:x+w]==i
        yy,xx=np.where(roimask)
        if len(xx)<4:continue
        eig=np.linalg.eigvalsh(np.cov(np.stack([xx,yy]).astype(float)))
        aspect=float(np.sqrt(max(1.e-8,eig[-1])/max(1.e-8,eig[0])))
        if aspect<min_elongation:continue
        count+=1;lab[y:y+h,x:x+w][roimask]=count
    return lab


def split():
    with (ROOT/'solar_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    return {s:[r for r in rows if r['split']==s] for s in ['train','tune','holdout']}


def evaluate_rows(net,rows,configs):
    tallies=[[] for _ in configs]
    for j,row in enumerate(rows):
        with np.load(CACHE/row['cache_file']) as d:im=d['im'];views=d['inst']
        probability=predict(net,im)
        for k,kwargs in enumerate(configs):
            mask=instance_map(probability,im,**kwargs)
            for view in views:tallies[k].append(pq_counts(view,mask))
        if (j+1)%20==0:print('PQ evaluation',j+1,'/',len(rows),flush=True)
    return [pooled_pq(k) for k in tallies]


def eval_model():
    torch.set_num_threads(4);net=read_model();records=split()
    configs=[{'threshold':thr,'min_area':area,'min_elongation':1.2} for thr in [.4,.5,.6,.7,.8] for area in [12,35,70]]
    tune=evaluate_rows(net,records['tune'],configs)
    j=max(range(len(tune)),key=lambda j:tune[j]['pq'])
    best=configs[j];result=evaluate_rows(net,records['holdout'],[best])[0]
    report={'selection':{'config':best,'tune_pq':tune[j],'n_tune':len(records['tune'])},
            'date_disjoint_audit':{'pooled_pq':result,'n_holdout':len(records['holdout'])},
            'all_tune_pq':[{**configs[k],'pq':v['pq']} for k,v in enumerate(tune)],
            'scope':'512x512 downsampled original real MAGFiLO train GT; NOT official Kaggle leaderboard score',
            'test_gt_used':False}
    (RESULT/'SOLAR_REAL_LOCAL_PQ_REPORT.json').write_text(json.dumps(report,indent=2))
    print('LOCAL REAL PQ:',json.dumps({'best':best,'tune':tune[j]['pq'],'audit':result['pq']}),flush=True)
    return best


def upscale_rle(mask,scale=4):
    m=np.asarray(mask,dtype=bool);h,w=m.shape;H=h*scale;total=H*w*scale
    columns=np.where(m.any(axis=0))[0]
    if not len(columns):return counts_to_string([total])
    intervals=[]
    for col in columns:
        yy=np.flatnonzero(m[:,col]);breaks=np.where(np.diff(yy)>1)[0]+1
        for run in np.split(yy,breaks):
            a=int(run[0]*scale);height=int(len(run)*scale)
            for t in range(scale):
                s=int((col*scale+t)*H+a);intervals.append((s,s+height))
    intervals.sort();counts=[];pos=0
    for start,end in intervals:
        if start==pos and len(counts)>=2:counts[-1]+=end-start
        else:counts.extend([start-pos,end-start])
        pos=end
    if pos<total:counts.append(total-pos)
    assert sum(counts)==total
    return counts_to_string(counts)


def run_test(config):
    net=read_model();csvout=RESULT/'SOLAR_FILAMENT_REAL_180_TEST_KAGGLE_CANDIDATE.csv'
    with zipfile.ZipFile(Z2) as z,csvout.open('w',newline='') as f:
        paths=sorted(n for n in z.namelist() if '/test/test_images/' in n and n.endswith('.jpeg'))
        assert len(paths)==180
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);counts=[]
        for j,name in enumerate(paths):
            im=cv2.imdecode(np.frombuffer(z.read(name),np.uint8),cv2.IMREAD_GRAYSCALE)
            assert im.shape==(2048,2048)
            small=cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
            labels=instance_map(predict(net,small),small,**config)
            n=int(labels.max());counts.append(n)
            for k in range(1,n+1):writer.writerow([Path(name).stem+'_'+str(k),upscale_rle(labels==k)])
            if (j+1)%30==0:print('REAL Kaggle test',j+1,'/180; predicted',sum(counts),'filaments',flush=True)
    sha=hashlib.sha256(csvout.read_bytes()).hexdigest()
    report={'images':len(counts),'predictions':sum(counts),'empty_images':counts.count(0),
            'median_instances_per_image':float(np.median(counts)),'filename':csvout.name,
            'sha256':sha,'file_bytes':csvout.stat().st_size,'full_resolution':[2048,2048],
            'official_submission':False,'note':'Unsubmitted Kaggle-format CSV, no hidden test GT used'}
    (RESULT/'SOLAR_KAGGLE_TEST_AUDIT.json').write_text(json.dumps(report,indent=2))
    print('SOLAR CSV FINISHED',json.dumps(report),flush=True)


def selftests():
    rng=np.random.default_rng(2026)
    for shape in [(7,4),(16,12)]:
      for _ in range(25):
        m=rng.random(shape)>.72
        expected=np.repeat(np.repeat(m,4,axis=0),4,axis=1).astype(np.uint8)
        rle=upscale_rle(m)
        assert encode(expected)==rle and np.array_equal(decode(rle,expected.shape),expected)
    print('RLE exact row/column and count tests passed',flush=True)

if __name__=='__main__':
    selftests();best=eval_model();run_test(best)
