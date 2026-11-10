"""Competition-resolution tiled U-Net inference and instances from trained weights.

Loads *only* weights trained from official MAGFiLO 2026 TRAIN images.
No test-time ground truth. Tunable threshold/area/aspect thresholds must be
picked using date-disjoint official training-only validation, not public test.
"""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from .train_unet import FilamentUNet
from .segment import disk_mask
from .coco_rle import encode


def windows(dim,tile,stride):
    if tile<16 or stride<=0 or stride>tile or dim<tile:raise ValueError('invalid tile/stride or image smaller than tile')
    starts=list(range(0,dim-tile+1,stride))
    if starts[-1]!=dim-tile:starts.append(dim-tile)
    return starts


def cosine_weights(tile):
    # Positive at edges: no divide-by-zero on borders, and smooth overlaps.
    t=np.linspace(0,np.pi,tile,dtype=np.float32)
    w=.08+.92*np.sin(t)**2
    return np.outer(w,w).astype(np.float32)


def tiled_probabilities(model,image,tile=384,stride=256,device='cpu',batch_size=4,flip_ensemble=False):
    gray=np.asarray(image,dtype=np.uint8)
    if gray.ndim!=2:raise ValueError('grayscale image expected')
    ys=windows(gray.shape[0],tile,stride);xs=windows(gray.shape[1],tile,stride)
    coords=[(y,x) for y in ys for x in xs]
    blend=np.zeros_like(gray,dtype=np.float32)
    weight=np.zeros_like(gray,dtype=np.float32)
    w=cosine_weights(tile)
    model.eval();torch.set_grad_enabled(False)
    try:
        for start in range(0,len(coords),batch_size):
            chunk=coords[start:start+batch_size]
            im=np.stack([gray[y:y+tile,x:x+tile].astype(np.float32)/255 for y,x in chunk])[:,None]
            tensor=torch.from_numpy(im).to(device)
            probability=torch.sigmoid(model(tensor)).cpu().numpy()[:,0]
            if flip_ensemble:
                flipped=torch.from_numpy(np.ascontiguousarray(im[:,:,:,::-1])).to(device)
                p2=torch.sigmoid(model(flipped)).cpu().numpy()[:,0,:,::-1]
                probability=(probability+p2)/2
            for (y,x),p in zip(chunk,probability):
                blend[y:y+tile,x:x+tile]+=p*w
                weight[y:y+tile,x:x+tile]+=w
    finally:
        torch.set_grad_enabled(True)
    return blend/np.maximum(weight,1e-8)


def probs_to_instances(prob,original_image,threshold=.45,min_area=70,min_elongation=1.7,bridge_px=3):
    gray=np.asarray(original_image,dtype=np.uint8)
    if prob.shape!=gray.shape or not np.isfinite(prob).all():raise ValueError('probability/image shape mismatch')
    solar_disk,_=disk_mask(gray)
    m=np.uint8((prob>=threshold)&solar_disk)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(bridge_px,bridge_px)))
    n,cc,stats,_=cv2.connectedComponentsWithStats(m,8)
    labels=np.zeros_like(cc,dtype=np.int32);inst=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<min_area:continue
        roi=(cc[y:y+h,x:x+w]==k)
        ry,rx=np.nonzero(roi)
        if len(rx)<4:continue
        eigenvalues=np.linalg.eigvalsh(np.cov(np.stack([rx,ry]).astype(float)))
        elong=float(np.sqrt(max(eigenvalues[-1],1e-8)/max(eigenvalues[0],1e-8)))
        if elong<min_elongation:continue
        newid=len(inst)+1
        labels[y:y+h,x:x+w][roi]=newid
        inst.append({'id':newid,'bbox':[x,y,w,h],'area':area,'elongation':elong,
                     'mean_predicted_confidence':float(np.mean(prob[y:y+h,x:x+w][roi]))})
    return labels,inst


def load_model(checkpoint,device='cpu'):
    # Strict load on a file created by this project's own training script.
    # Never unpickle untrusted model files from public forums.
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=FilamentUNet(base=int(ckpt['base']))
    model.load_state_dict(ckpt['state_dict'],strict=True)
    return model.to(device).eval()


def kaggle_predict(image_paths,checkpoint,output,device='cpu',threshold=.45,min_area=70,batch_size=4,flip_ensemble=False):
    model=load_model(checkpoint,device)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for path in image_paths:
            p=Path(path);image=np.asarray(Image.open(p).convert('L'))
            if image.shape!=(2048,2048):raise ValueError('official 2048x2048 test image required')
            prob=tiled_probabilities(model,image,device=device,batch_size=batch_size,flip_ensemble=flip_ensemble)
            labels,inst=probs_to_instances(prob,image,threshold,min_area)
            for row in inst:
                rle=encode(np.asfortranarray(np.uint8(labels==row['id'])))
                writer.writerow([p.stem+'_'+str(row['id']),rle])
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--images',nargs='+',required=True)
    parser.add_argument('--weights',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threshold',type=float,default=.45)
    parser.add_argument('--batch-size',type=int,default=4);parser.add_argument('--tta',action='store_true')
    args=parser.parse_args()
    print(kaggle_predict(args.images,args.weights,args.output,args.device,args.threshold,batch_size=args.batch_size,flip_ensemble=args.tta))
if __name__=='__main__':main()
