"""Real MAGFiLO test-time augmentation ablation with date-separated tuning.

Only compare normal U-Net vs reflection-averaged U-Net; no organizer test
labels are used. Do not promote the new method unless tuning PQ improves.
"""
import csv,json,zipfile,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from solar_real import ROOT,CACHE,RESULT,Z2,read_model,predict
from solar_eval import split,instance_map,upscale_rle
from solar_filament.metrics import pq_counts,pooled_pq


def predict_tta(net,image):
    img=np.ascontiguousarray(image)
    flips=[img,img[:,::-1],img[::-1,:],img[::-1,::-1]]
    imgs=np.stack([np.ascontiguousarray(x) for x in flips]).astype(np.float32)[:,None]/255.
    with torch.inference_mode():
        p=torch.sigmoid(net(torch.from_numpy(imgs))).numpy()[:,0]
    q=(p[0]+p[1][:,::-1]+p[2][::-1,:]+p[3][::-1,::-1])*.25
    return q


def pooled(net,rows,configs):
    normal=[[] for _ in configs];aug=[[] for _ in configs]
    for i,row in enumerate(rows):
        with np.load(CACHE/row['cache_file']) as d:image=d['im'];gt=d['inst']
        probs=[predict(net,image),predict_tta(net,image)]
        for k,config in enumerate(configs):
            for j,proba in enumerate(probs):
                mask=instance_map(proba,image,**config)
                for lab in gt:(normal if j==0 else aug)[k].append(pq_counts(lab,mask))
        if (i+1)%15==0:print('TTA VALID',i+1,'/',len(rows),flush=True)
    return [[pooled_pq(x) for x in table] for table in (normal,aug)]


def evaluate():
    net=read_model();records=split()
    base=json.loads((RESULT/'SOLAR_REAL_LOCAL_PQ_REPORT.json').read_text())['selection']['config']
    cfg=[base]+[{'threshold':t,'min_area':a,'min_elongation':1.2} for t in (.5,.6,.7,.8) for a in (20,45,75)]
    scores=pooled(net,records['tune'],cfg)
    winner=max(((scores[j][i]['pq'],j,i) for j in range(2) for i in range(len(cfg))))
    top,j,i=winner
    flag=['original','flipped_average'][j];selected=cfg[i]
    print('TTA selected',flag,selected,top,flush=True)
    # Only the tuning subset is used to select model method and hyperparams.
    held=pooled(net,records['holdout'],[selected])
    result={'selection':'train-date-disjoint 63-image tuning subset',
            'method':flag,'parameters':selected,'tuning_pq':top,
            'holdout_baseline_pq':held[0][0]['pq'],'holdout_tta_pq':held[1][0]['pq'],
            'holdout_selected_pq':held[j][0]['pq'],
            'test_ground_truth_used':False,'comment':'Local downsampled 512px PQ; 62-image date-separated audit; not official Kaggle score.'}
    (RESULT/'SOLAR_TTA_COMPARISON.json').write_text(json.dumps(result,indent=2));return result


def test_180_method(selection):
    if selection['method']!='flipped_average':
        print('TTA did not win tuning; original candidate remains preferred',flush=True);return None
    net=read_model();csvout=RESULT/'SOLAR_180_TEST_TTA_ALTERNATE.csv'
    config=selection['parameters'];counts=[]
    with zipfile.ZipFile(Z2) as z,csvout.open('w',newline='') as f:
        files=sorted(n for n in z.namelist() if '/test/test_images/' in n and n.endswith('.jpeg'))
        assert len(files)==180
        out=csv.writer(f);out.writerow(['filament_id','segmentation_rle'])
        for j,name in enumerate(files):
            im=cv2.imdecode(np.frombuffer(z.read(name),np.uint8),cv2.IMREAD_GRAYSCALE)
            x=cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
            mask=instance_map(predict_tta(net,x),x,**config)
            counts.append(int(mask.max()))
            for v in range(1,int(mask.max())+1):out.writerow([Path(name).stem+'_'+str(v),upscale_rle(mask==v)])
            if (j+1)%30==0:print('TTA TEST',j+1,'/180',flush=True)
    audit={'images':180,'instances':sum(counts),'images_without_prediction':counts.count(0),
           'sha256':hashlib.sha256(csvout.read_bytes()).hexdigest(),
           'source':'Kaggle unlabeled test image ZIP, original input 2048px, downscaled 512px model, 4 flipped TTA predictions',
           'not_official_submission':True}
    (RESULT/'SOLAR_TTA_TEST_CSV_AUDIT.json').write_text(json.dumps(audit,indent=2));return audit

if __name__=='__main__':
    result=evaluate()
    print(json.dumps(result,indent=2),flush=True)
    print(test_180_method(result),flush=True)
