"""Organizer TRAIN dataset integrity and contamination guards.

No label downloads. Requires user to accept competition rules, obtain
organizer-provided MAGFiLO_1.0_Annotations_kaggle2026_train.json, images.
Public original 2024 release may contain Kaggle 2026 test labels; refuse to
substitute that release for the organizer's train-only annotations.
"""
from pathlib import Path
import json,hashlib
from PIL import Image
from .dataset import group_coco,date_holdout

REQUIRED_ANNOTATION_BASENAME='MAGFiLO_1.0_Annotations_kaggle2026_train.json'


def verify_official_train(root,annotation_path,check_hashes=True):
    base=Path(root);anns=Path(annotation_path)
    if anns.name!=REQUIRED_ANNOTATION_BASENAME:
        raise ValueError('REFUSED: only the named Kaggle 2026 TRAIN annotations are allowed; never complete MAGFiLO 2024 test labels')
    groups=group_coco(anns)
    tr,val=date_holdout(groups)
    if not tr or not val:raise ValueError('need independent train and validation dates')
    if set(x[:8] for x in tr)&set(x[:8] for x in val):
        raise ValueError('same-day train/val contamination')
    errors=[];meta=[]
    for name,annotations in sorted(groups.items()):
        path=base/name
        if not path.is_file():
            errors.append(f'missing {name}');continue
        with Image.open(path) as im:
            if im.size!=(2048,2048):errors.append(f'bad dimensions: {name} {im.size}')
        h=hashlib.sha256(path.read_bytes()).hexdigest() if check_hashes else None
        meta.append({'file':name,'annotators':len(annotations),
                     'instances_by_annotator':[len(view[1]) for view in annotations],
                     'sha256':h})
    return {'source':str(anns),'official_source_required':True,
            'training_physical_observations':len(groups),
            'independent_annotator_views':sum(len(v) for v in groups.values()),
            'observation_dates':len(set(n[:8] for n in groups)),
            'train_dates':len(set(n[:8] for n in tr)),
            'holdout_dates':len(set(n[:8] for n in val)),
            'date_disjoint':True,'files':meta,'errors':errors,
            'ready_for_official_train':not bool(errors),
            'caveat':'passing the preflight does not establish validated PQ or eligibility'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--annotation',required=True)
    p.add_argument('--out',required=True);p.add_argument('--skip-hash',action='store_true')
    a=p.parse_args()
    data=verify_official_train(a.images,a.annotation,not a.skip_hash)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!='files'},indent=2))
    if not data['ready_for_official_train']:raise SystemExit(2)
