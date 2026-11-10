"""Structure-sensitive H-alpha filament candidates from *images only*.

A controlled ablation companion to the original morphological black-hat method.
Retains elongated dark ridges, filters round sunspots, bridges small gaps,
with no use of test labels and no claim of organzier-validated performance.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from .segment import disk_mask


@dataclass(frozen=True)
class Parameters:
    response_percentile:float=92.0
    line_min_area:int=25
    min_elongation:float=1.7
    bridge_length:int=9
    min_contrast:float=2.5
    multi_scale:tuple=(7,15,29)


def _elongation(component_pixels):
    yy,xx=np.nonzero(component_pixels)
    if len(yy)<3:return 0.
    covariance=np.cov(np.stack((xx.astype(float),yy.astype(float))))
    eig=np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(eig[-1],1e-8)/max(eig[0],1e-8)))


def _line_connect(mask,length=7):
    """Orientation-aware closing: candidate thin gaps need multi-direction evidence.

    A union of all closings risks merging separate filaments. Instead a bridge
    is accepted only if it is supported by at least two orientations and the
    new pixel lies within 3px of an existing ridge.
    """
    m=np.uint8(mask)
    n=max(3,int(length)|1)
    directions=[]
    for ang in range(0,180,30):
        kernel=np.zeros((n,n),np.uint8)
        center=n//2;dx=int(round(center*np.cos(np.deg2rad(ang))))
        dy=int(round(center*np.sin(np.deg2rad(ang))))
        cv2.line(kernel,(center-dx,center-dy),(center+dx,center+dy),1,1)
        directions.append(cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel))
    support=sum((x>0).astype('uint8') for x in directions)
    original_near=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    return (m>0)|((support>=2)&original_near)


def segment(gray,params=Parameters()):
    x=np.asarray(gray)
    if x.ndim==3:x=cv2.cvtColor(x.astype('uint8'),cv2.COLOR_RGB2GRAY)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite 2D grayscale required')
    if not (70<=params.response_percentile<=99.95):raise ValueError('response percentile out of range')
    if x.max()>255 or x.min()<0:
        lo,hi=np.percentile(x,[.1,99.9]);x=np.uint8(np.clip((x-lo)/max(hi-lo,1e-6)*255,0,255))
    else:x=np.asarray(x,dtype=np.uint8)
    disk,geom=disk_mask(x)
    # Illumination inversion; black-hat captures dark objects at 3 scales.
    background=cv2.GaussianBlur(x,(0,0),max(12,x.shape[0]/18))
    flat=np.uint8(np.clip(128+x.astype('float32')-background,0,255))
    contrast=np.zeros_like(x,dtype='float32')
    for k in params.multi_scale:
        size=2*(int(k)//2)+1
        hat=cv2.morphologyEx(flat,cv2.MORPH_BLACKHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
        contrast=np.maximum(contrast,hat.astype('float32'))
    blur=gaussian_filter(contrast,1.0)
    inner=disk & (contrast>params.min_contrast)
    if not np.any(inner):
        return np.zeros_like(x,dtype=np.int32),{'detected':0,'threshold':None,'disk':geom}
    threshold=float(max(params.min_contrast,np.percentile(blur[inner],params.response_percentile)))
    proposed=(blur>=threshold)&disk
    # Black-hat has positive response for sunspots too; component eigen-ratio
    # is used to filter round objects. Closing is intentionally conservative.
    proposed=_line_connect(proposed,params.bridge_length)&disk
    n,cc,stats,centers=cv2.connectedComponentsWithStats(np.uint8(proposed),8)
    labels=np.zeros(x.shape,np.int32);objects=[]
    for k in range(1,n):
        x0,y0,w,h,area=map(int,stats[k])
        if area<params.line_min_area:continue
        roi=(cc[y0:y0+h,x0:x0+w]==k)
        elong=_elongation(roi)
        if elong<params.min_elongation:continue
        idx=len(objects)+1
        labels[y0:y0+h,x0:x0+w][roi]=idx
        objects.append({'id':idx,'area':area,'bbox':[x0,y0,w,h],
                        'elongation':round(elong,3)})
    return labels,{'detected':len(objects),'threshold':threshold,
                   'disk':geom,'parameters':params.__dict__,'instances':objects,
                   'limitations':'untrained, must evaluate on official date-isolated MAGFiLO train masks'}
