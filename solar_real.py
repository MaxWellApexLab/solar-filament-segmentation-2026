"""Recover and advance real MAGFiLO segmentation from two uploaded organizer ZIPs.

No use of unlabeled Kaggle test images during training or model selection.
Date-disjoint train/tune/audit separation and pooled annotator-wise local PQ.
"""
import csv,hashlib,json,random,time,zipfile,os
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'recover/PrizeStrike_AeroCore_ScientificUpgrade_20261104'))
from solar_filament.train_unet import FilamentUNet
from solar_filament.metrics import pq_counts,pooled_pq
from solar_filament.coco_rle import encode,decode
Z1=Path(os.environ.get('SOLAR_ZIP1','/mnt/data/00.zip'));Z2=Path(os.environ.get('SOLAR_ZIP2','/mnt/data/MAGFiLO_1.0_Kaggle_2026.zip'))
CACHE=ROOT/'solar_cache';RESULT=ROOT/'results';RESULT.mkdir(exist_ok=True)
ANNO='MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'


def groups():
 with zipfile.ZipFile(Z2) as z:data=json.loads(z.read(ANNO))
 pics={x['id']:x for x in data['images']};anns=defaultdict(list);out=defaultdict(list)
 for a in data['annotations']:anns[a['image_id']].append(a)
 for id,im in pics.items():out[Path(im['file_name']).name].append(anns[id])
 return dict(out)


def cache():
 CACHE.mkdir(exist_ok=True)
 g=groups();names=sorted(g);dates=sorted({n[:8] for n in names})
 rng=np.random.default_rng(2026);rng.shuffle(dates);val=set(dates[:int(len(dates)*.18)])
 held=sorted(val);random.Random(4251).shuffle(held);hold=set(held[::2])
 with zipfile.ZipFile(Z1) as z1,zipfile.ZipFile(Z2) as z2:
  files1={Path(n).name:n for n in z1.namelist() if n.endswith('.jpeg')};files2={Path(n).name:n for n in z2.namelist() if '/train/train_images/' in n and n.endswith('.jpeg')}
  assert len(files1)==589 and len(files2)==118 and len(g)==707
  rows=[]
  for i,name in enumerate(names):
   output=CACHE/(Path(name).stem+'.npz')
   if not output.exists():
    z,n=(z1,files1[name]) if name in files1 else (z2,files2[name])
    im=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),cv2.IMREAD_GRAYSCALE)
    assert im.shape==(2048,2048)
    small=cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
    versions=[]
    for anns in g[name]:
     lab=np.zeros((512,512),np.uint16)
     index=0
     for ann in anns:
      seg=ann.get('segmentation',[])
      if not isinstance(seg,list):continue
      m=np.zeros((512,512),np.uint8)
      for poly in seg:
       pts=np.asarray(poly,float).reshape(-1,2)
       if len(pts)>=3:cv2.fillPoly(m,[np.rint(pts/4).astype(np.int32)],1)
      if m.any():
       index+=1;lab[(m>0)&(lab==0)]=index
     versions.append(lab)
    np.savez_compressed(output,im=small,inst=np.stack(versions))
   split='train' if name[:8] not in val else ('holdout' if name[:8] in hold else 'tune')
   rows.append({'name':name,'date':name[:8],'split':split,'cache_file':output.name,'views':len(g[name])})
   if i%150==0:print('prepared image',i,'/707',flush=True)
 with (ROOT/'solar_manifest.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,rows[0].keys());wr.writeheader();wr.writerows(rows)
 counts={s:sum(r['split']==s for r in rows) for s in ['train','tune','holdout']}
 summary={'splits':counts,'total_images':707,'annotator_views':sum(r['views'] for r in rows),'test_used':False}
 (RESULT/'SOLAR_DATA_AUDIT.json').write_text(json.dumps(summary,indent=2));print('PREPARE FINISHED',summary,flush=True)
 return rows

class Crops(Dataset):
 def __init__(self,rows,epoch):self.rows=rows;self.epoch=epoch
 def __len__(self):return len(self.rows)*2
 def __getitem__(self,i):
  row=self.rows[i%len(self.rows)];rng=np.random.default_rng(1729+i+7_912_001*self.epoch)
  with np.load(CACHE/row['cache_file']) as d:
   raw=d['im'];lab=d['inst'][int(rng.integers(len(d['inst'])))]>0
  yy,xx=np.where(lab)
  if len(xx) and rng.random()<.88:
   j=int(rng.integers(len(xx)));x,y=int(xx[j]),int(yy[j])
  else:x,y=int(rng.integers(512)),int(rng.integers(512))
  a=int(np.clip(x-128,0,256));b=int(np.clip(y-128,0,256))
  x=raw[b:b+256,a:a+256].astype(np.float32)/255.;y=lab[b:b+256,a:a+256].astype(np.float32)
  if rng.random()<.5:x=np.fliplr(x);y=np.fliplr(y)
  if rng.random()<.5:x=np.flipud(x);y=np.flipud(y)
  x=np.clip(x*float(rng.uniform(.8,1.2))+float(rng.uniform(-.065,.065)),0,1)
  return torch.from_numpy(np.ascontiguousarray(x[None])),torch.from_numpy(np.ascontiguousarray(y[None]))

def lossfn(p,y):
 bce=torch.nn.functional.binary_cross_entropy_with_logits(p,y,pos_weight=torch.tensor(4.,device=p.device))
 q=torch.sigmoid(p);d=(2*(q*y).sum((2,3))+1)/(q.sum((2,3))+y.sum((2,3))+1)
 return .6*bce+.4*(1-d.mean())

def predict(net,im):
 with torch.inference_mode():return torch.sigmoid(net(torch.from_numpy(im[None,None].astype(np.float32)/255.))).numpy()[0,0]

def val_loss(net,rows):
 losses=[];net.eval()
 for row in rows:
  with np.load(CACHE/row['cache_file']) as d:
   im=d['im'];labs=d['inst']
  with torch.inference_mode():
   z=net(torch.from_numpy(im[None,None].astype(np.float32)/255.))
   for y in labs:losses.append(float(lossfn(z,torch.from_numpy((y>0).astype(np.float32)[None,None]))))
 return float(np.mean(losses))

def train(epochs=10):
 torch.set_num_threads(4);torch.manual_seed(20261104);np.random.seed(20261104)
 with (ROOT/'solar_manifest.csv').open() as f:rows=list(csv.DictReader(f))
 sets={s:[r for r in rows if r['split']==s] for s in ['train','tune','holdout']}
 date_sets=[{r['date'] for r in sets[s]} for s in sets]
 assert len(set.union(*date_sets))==sum(len(x) for x in date_sets)
 net=FilamentUNet(base=8);opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.0001)
 hist=[];best=1e9
 for epoch in range(epochs):
  t=time.time();net.train();losses=[]
  for x,y in DataLoader(Crops(sets['train'],epoch),batch_size=6,shuffle=True,num_workers=0):
   opt.zero_grad(set_to_none=True);p=net(x);l=lossfn(p,y);l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();losses.append(float(l.detach()))
  tune=val_loss(net,sets['tune'])
  hist.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'tune_loss':tune,'elapsed_seconds':round(time.time()-t,1)})
  print('EPOCH',epoch+1,'tune',round(tune,4),flush=True)
  if tune<best:
   best=tune;torch.save({'state_dict':net.state_dict(),'base':8,'epoch':epoch+1,'train_size':len(sets['train'])},RESULT/'SOLAR_REAL_TRAINED_UNET.pt')
   # Durable checkpoint after every improved epoch.
   (RESULT/'SOLAR_TRAIN_HISTORY.json').write_text(json.dumps({'history':hist,'best_tune_loss':best},indent=2))
 print('TRAINING COMPLETE',best,flush=True)

def read_model():
 ck=torch.load(RESULT/'SOLAR_REAL_TRAINED_UNET.pt',weights_only=True,map_location='cpu');net=FilamentUNet(base=ck['base']);net.load_state_dict(ck['state_dict']);net.eval();torch.set_num_threads(4);return net

if __name__=='__main__':
 cache();train()
