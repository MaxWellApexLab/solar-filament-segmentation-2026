"""Official MAGFiLO Kaggle COCO annotation loader.

Multi-rater images sharing a physical filename MUST stay in the same split.
We select a random annotated view per item each epoch rather than merging
inconsistent opinions into fabricated ground truth. No Kaggle test labels read.
"""
from collections import defaultdict
from pathlib import Path
from PIL import Image
import cv2,json,random
import numpy as np
import torch
from torch.utils.data import Dataset
from .coco_rle import decode


def masks_from_coco(annotations,shape):
    h,w=shape;instances=np.zeros((h,w),np.int32)
    for i,ann in enumerate(annotations,1):
        seg=ann.get('segmentation')
        mask=np.zeros((h,w),np.uint8)
        if isinstance(seg,list):
            for poly in seg:
                points=np.asarray(poly,dtype=float).reshape(-1,2)
                if len(points)>=3:
                    cv2.fillPoly(mask,[np.round(points).astype(np.int32)],1)
        elif isinstance(seg,dict):
            cnts=seg['counts']
            if isinstance(cnts,str):mask=decode(cnts,tuple(seg.get('size',shape)))
            elif isinstance(cnts,list):
                from .coco_rle import counts_to_string
                mask=decode(counts_to_string(cnts),tuple(seg.get('size',shape)))
        if np.any(mask):instances[(mask>0)&(instances==0)]=i
    return instances


def group_coco(annotation_file):
    data=json.loads(Path(annotation_file).read_text())
    images={im['id']:im for im in data['images']}
    groups=defaultdict(list)
    by_image=defaultdict(list)
    for ann in data['annotations']:by_image[ann['image_id']].append(ann)
    for id_,im in images.items():
        name=Path(im['file_name']).name
        groups[name].append((im,by_image[id_]))
    return dict(groups)


def date_holdout(groups,ratio=.2,seed=2026):
    """Use observation DATE split; not file-level if multiple observations same day."""
    names=list(groups)
    dates=sorted(set(n[:8] for n in names))
    rng=random.Random(seed);rng.shuffle(dates)
    val=set(dates[:max(1,round(len(dates)*ratio))])
    train=[name for name in names if name[:8] not in val]
    holdout=[name for name in names if name[:8] in val]
    return train,holdout

class CropDataset(Dataset):
    def __init__(self,image_root,annotation_file,names=None,crop_size=384,patches_per_image=5,seed=2026):
        self.path=Path(image_root);self.groups=group_coco(annotation_file)
        self.names=list(names if names is not None else self.groups)
        self.crop=int(crop_size);self.repeat=int(patches_per_image)
        self.seed=int(seed)
        if not self.names:raise ValueError('empty image list')
    def __len__(self):return len(self.names)*self.repeat
    def __getitem__(self,index):
        name=self.names[index//self.repeat]
        view_id=index%self.repeat
        rng=np.random.default_rng(self.seed+index)
        im,anns=self.groups[name][int(rng.integers(len(self.groups[name])))]
        raw=np.asarray(Image.open(self.path/name).convert('L'),dtype=np.uint8)
        gt=(masks_from_coco(anns,raw.shape)>0).astype(np.float32)
        h,w=raw.shape;c=self.crop
        if h<c or w<c:raise ValueError('image smaller than crop')
        yy,xx=np.nonzero(gt)
        if len(xx) and rng.uniform()<.8:
            k=int(rng.integers(len(xx)));x,y=int(xx[k]),int(yy[k])
        else:x,y=int(rng.integers(w)),int(rng.integers(h))
        x0=int(np.clip(x-c//2,0,w-c));y0=int(np.clip(y-c//2,0,h-c))
        image=raw[y0:y0+c,x0:x0+c].astype(np.float32)/255.
        mask=gt[y0:y0+c,x0:x0+c]
        # Diverse illumination without augmenting validation automatically.
        image=np.clip((image-.5)*float(rng.uniform(.8,1.2))+.5+float(rng.uniform(-.08,.08)),0,1)
        if rng.uniform()<.5:image=np.fliplr(image);mask=np.fliplr(mask)
        if rng.uniform()<.5:image=np.flipud(image);mask=np.flipud(mask)
        return torch.from_numpy(np.ascontiguousarray(image[None])).float(),torch.from_numpy(np.ascontiguousarray(mask[None])).float()
