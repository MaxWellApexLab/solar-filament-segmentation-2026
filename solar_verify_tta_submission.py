"""Independent complete COCO-RLE/schema/test-image audit for TTA candidate."""
from pathlib import Path
import csv,hashlib,json,random,re,zipfile
from solar_real import RESULT,Z2
from solar_filament.coco_rle import string_to_counts,decode


def verify(path=None):
    source=Path(path) if path else RESULT/'SOLAR_180_TEST_TTA_ALTERNATE.csv'
    declared=json.loads((RESULT/'SOLAR_TTA_TEST_CSV_AUDIT.json').read_text())
    with zipfile.ZipFile(Z2) as z:
        valid={Path(n).stem for n in z.namelist() if '/test/test_images/' in n and n.endswith('.jpeg')}
    assert len(valid)==180
    seen=set();count_by_image={x:[] for x in valid};mask_records=[]
    with source.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Unexpected Kaggle submission header')
        for row in reader:
            name=row['filament_id']
            m=re.fullmatch(r'(.+)_([1-9][0-9]*)',name)
            if not m or m.group(1) not in valid or name in seen:
                raise ValueError(f'Invalid, repeated or unknown filament ID {name}')
            seen.add(name)
            count_by_image[m.group(1)].append(int(m.group(2)))
            run=string_to_counts(row['segmentation_rle'])
            if not run or sum(run)!=2048*2048 or any(v<0 for v in run) or sum(run[1::2])<1:
                raise ValueError(f'Invalid RLE for {name}')
            mask_records.append(row['segmentation_rle'])
    for name,v in count_by_image.items():
        if v!=list(range(1,len(v)+1)):raise ValueError(f'Nonconsecutive instance IDs {name}')
    sample=random.Random(20261104).sample(mask_records,min(30,len(mask_records)))
    for encoded in sample:
        image=decode(encoded,(2048,2048))
        assert image.shape==(2048,2048) and image.dtype.kind in 'bui' and image.any()
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==declared['sha256']
    result={'source_file':source.name,'sha256':sha,'valid_rows':len(seen),
      'expected_test_images':len(valid),'empty_prediction_images':sum(not v for v in count_by_image.values()),
      'actual_decoded_2048_masks':len(sample),'all_rle_counts_sum_4194304':True,
      'id_unique_and_source_verified':True,'test_gt_used':False,
      'organizer_score':None,'caveat':'Kaggle-compatible *candidate*, not an organizer-accepted submission.'}
    (RESULT/'SOLAR_TTA_ALL180_INDEPENDENT_QA.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':print(json.dumps(verify(),indent=2))
