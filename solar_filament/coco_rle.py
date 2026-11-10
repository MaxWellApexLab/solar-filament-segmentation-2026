"""Pure Python COCO compressed RLE implementation (mask_api wire format).

Runs counted from zero-valued pixels along *column-major* flattening. The
compressed counts string is the COCO varint with two-back difference coding;
we validate against decoded mask with asymmetric fixtures. If pycocotools is
available, compare on random masks before submitting competition CSV.
"""
import numpy as np


def mask_to_counts(mask):
    m=np.asarray(mask,dtype=np.uint8)
    if m.ndim!=2:raise ValueError('mask must be 2D')
    flat=m.flatten(order='F')
    if np.any((flat!=0)&(flat!=1)):raise ValueError('binary mask required')
    # State changes start with position zero if the first pixel is 1.
    changes=np.flatnonzero(np.diff(flat.astype(np.int16)))+1
    breaks=np.r_[0,changes,flat.size]
    counts=np.diff(breaks).astype(int).tolist()
    if flat.size and flat[0]!=0:counts.insert(0,0)
    return counts


def counts_to_string(counts):
    out=[]
    for i,v in enumerate(counts):
        x=int(v)
        if i>2:x-=int(counts[i-2])
        while True:
            c=x&0x1f
            x>>=5
            more=(x!=-1) if (c&0x10) else (x!=0)
            if more:c|=0x20
            c+=48
            out.append(chr(c))
            if not more:break
    return ''.join(out)


def string_to_counts(s):
    counts=[];i=0
    while i<len(s):
        x=0;k=0
        while True:
            c=ord(s[i])-48
            x|=(c&0x1f)<<(5*k)
            more=c&0x20
            i+=1;k+=1
            if not more:
                if c&0x10:x|=-(1<<(5*k))
                break
        if len(counts)>2:x+=counts[-2]
        counts.append(x)
    return counts


def encode(mask):
    return counts_to_string(mask_to_counts(mask))


def decode(s,shape):
    counts=string_to_counts(s)
    flat=np.zeros(int(np.prod(shape)),np.uint8)
    index=0;is_one=False
    for v in counts:
        if v<0 or index+v>len(flat):raise ValueError('corrupt COCO compressed RLE')
        if is_one:flat[index:index+v]=1
        index+=v;is_one=not is_one
    if index!=len(flat):raise ValueError('decoded COCO RLE pixel count mismatch')
    return flat.reshape(shape,order='F')
