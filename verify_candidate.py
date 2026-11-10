"""Validate all genuine Solar candidate COCO mask rows and test names."""
import csv,json,hashlib,re,zipfile,random
from pathlib import Path
from solar_filament.coco_rle import string_to_counts,decode
from solar_source import OfficialTestImages


def verify(candidate,source_kaggle_zip,decode_count=25):
    candidate=Path(candidate);known={}
    with OfficialTestImages(source_kaggle_zip) as source:
        known={name:[] for name in source.paths}
    if len(known)!=180:raise ValueError('expected 180 test images')
    all_rle=[]
    with candidate.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('invalid header')
        for row in reader:
            m=re.fullmatch(r'(.+)_([1-9][0-9]*)',row['filament_id'])
            if not m or m[1] not in known:raise ValueError('foreign label '+row['filament_id'])
            known[m[1]].append(int(m[2]))
            a=string_to_counts(row['segmentation_rle'])
            if sum(a)!=4194304 or any(v<0 for v in a) or sum(a[1::2])<1:
                raise ValueError('invalid mask '+row['filament_id'])
            all_rle.append(row['segmentation_rle'])
    for name,ids in known.items():
        if ids!=list(range(1,len(ids)+1)):raise ValueError('duplicate/skip '+name)
    for rle in random.Random(2026).sample(all_rle,min(decode_count,len(all_rle))):
        if not decode(rle,(2048,2048)).any():raise ValueError('empty decoded mask')
    return {'images':180,'rows':len(all_rle),'empty_images':sum(not x for x in known.values()),
            'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'all_masks_have_correct_shape':True,'decoded_masks':min(decode_count,len(all_rle)),
            'official_score':None,'submitted':False}

if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1],sys.argv[2]),indent=2))
