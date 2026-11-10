"""CPU non-neural H-alpha filament-instance baseline.

True 2048x2048 GONG competition JPEGs ONLY. This is an untrained physical
image-processing baseline, NOT a trained or leaderboard-validated solution.

The solar disk mask is estimated from image luminance. A multiscale black top-hat
isolates dark regions; local contrast and elongation suppress sunspot-like blobs.
Parameters should be tuned on official train masks, not test labels.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def disk_mask(gray, limb_erosion=0.035):
    """Circular H-alpha disk approximation from non-background pixels."""
    if gray.ndim != 2:
        raise ValueError('2-D grayscale image required')
    h,w=gray.shape
    blurred=cv2.GaussianBlur(gray,(0,0),3)
    thresh=max(12.0,float(np.percentile(blurred,95))*.14)
    keep=np.uint8(blurred>thresh)
    num, labels, stats, centers=cv2.connectedComponentsWithStats(keep,8)
    if num<2:raise ValueError('No detectable sun disk')
    best=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    ys,xs=np.where(labels==best)
    if len(xs)<0.04*h*w:raise ValueError('No sizable sun disk')
    cx=(float(np.percentile(xs,5))+float(np.percentile(xs,95)))/2
    cy=(float(np.percentile(ys,5))+float(np.percentile(ys,95)))/2
    radius=min(float(np.percentile(xs,95)-np.percentile(xs,5)),
               float(np.percentile(ys,95)-np.percentile(ys,5)))/1.9
    yy,xx=np.ogrid[:h,:w]
    radial=((xx-cx)**2+(yy-cy)**2)<(radius*(1-limb_erosion))**2
    return radial, {'cx':round(cx,2),'cy':round(cy,2),'radius':round(radius,2)}


def segment(gray,scales=(9,19,35),threshold=21.,min_area=70,min_elongation=2.0):
    x=np.asarray(gray)
    if x.ndim==3:
        x=cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    if x.ndim!=2:raise ValueError('Must supply grayscale/RGB image')
    if not np.isfinite(x).all():raise ValueError('nonfinite image')
    if x.max()>255 or x.min()<0:
        lo,hi=np.percentile(x,[.1,99.9]);x=(255*np.clip((x-lo)/max(hi-lo,1e-6),0,1)).astype(np.uint8)
    else:x=x.astype(np.uint8)
    disk,geom=disk_mask(x)
    # Flatten low-frequency intensity variation, but preserve fine structures.
    light=cv2.GaussianBlur(x,(0,0),max(20.,min(x.shape)*.025))
    normalized=np.uint8(np.clip(x.astype(np.float32)-light.astype(np.float32)+128,0,255))
    response=np.zeros_like(normalized,dtype=np.float32)
    for scale in scales:
        k=2*(int(scale)//2)+1
        blackhat=cv2.morphologyEx(normalized,cv2.MORPH_BLACKHAT,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        response=np.maximum(response,blackhat.astype(np.float32))
    # Disk corrected contrast quantiles and top-hat strength.
    mask=np.uint8((response>threshold)&disk)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    n, components, stats, centroids=cv2.connectedComponentsWithStats(mask,8)
    labels=np.zeros(mask.shape,dtype=np.int32);results=[]
    for i in range(1,n):
        x0,y0,w,h,area=map(int,stats[i]);
        if area<min_area:continue
        # Elongation using second moments; reject circular sunspot-like blobs.
        iy,ix=np.where(components[y0:y0+h,x0:x0+w]==i)
        if len(ix)<3:continue
        mat=np.stack((ix.astype(float),iy.astype(float)))
        covariance=np.cov(mat)
        eig=np.linalg.eigvalsh(covariance)
        aspect=float(np.sqrt(max(eig[-1],1e-8)/max(eig[0],1e-8)))
        if aspect<min_elongation:continue
        nid=len(results)+1
        labels[y0:y0+h,x0:x0+w][components[y0:y0+h,x0:x0+w]==i]=nid
        results.append({'id':nid,'pixels':area,'elongation':round(aspect,2),
                        'bbox_xywh':[x0,y0,w,h],'mean_contrast':float(np.mean(response[components==i]))})
    return labels,{'disk':geom,'scale_px':list(scales),'threshold':float(threshold),'detected':len(results),'instances':results,
                   'limitation':'untrained baseline; check against official MAGFiLO labeled train data'}


def read(path):
    return np.asarray(Image.open(path).convert('L'))


def to_kaggle_csv(image_paths,output,threshold=21.,min_area=70):
    from .coco_rle import encode,decode
    import csv
    image_paths=[Path(p) for p in image_paths]
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['filament_id','segmentation_rle'])
        for img in image_paths:
            arr=read(img)
            if arr.shape!=(2048,2048):
                raise ValueError(f'{img.name}: challenge format requires 2048x2048')
            instances,_=segment(arr,threshold=threshold,min_area=min_area)
            for i in range(1,instances.max()+1):
                m=np.asfortranarray((instances==i).astype(np.uint8))
                counts=encode(m)
                assert np.array_equal(decode(counts,m.shape),m),'COCO RLE roundtrip mismatch'
                wr.writerow([img.stem+'_'+str(i),counts])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--images',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threshold',type=float,default=21);p.add_argument('--min-area',type=int,default=70)
    a=p.parse_args();print(to_kaggle_csv(a.images,a.output,a.threshold,a.min_area))
if __name__=='__main__':main()
