"""Check all 180 filenames, predicted IDs and sparse COCO RLE counts exactly."""
import csv,hashlib,json,re,zipfile,random
from pathlib import Path
from solar_real import RESULT,Z2
from solar_filament.coco_rle import string_to_counts,decode


def verify():
    p=RESULT/'SOLAR_FILAMENT_REAL_180_TEST_KAGGLE_CANDIDATE.csv'
    meta=json.loads((RESULT/'SOLAR_KAGGLE_TEST_AUDIT.json').read_text())
    with zipfile.ZipFile(Z2) as z:
        valid={Path(f).stem for f in z.namelist() if '/test/test_images/' in f and f.endswith('.jpeg')}
    assert len(valid)==180
    seen=set();counts={x:[] for x in valid};to_decode=[]
    with p.open(newline='',encoding='utf-8') as f:
        d=csv.DictReader(f);assert d.fieldnames==['filament_id','segmentation_rle']
        for row in d:
            ident=row['filament_id'];match=re.fullmatch(r'(.+)_([1-9][0-9]*)',ident)
            if not match or match[1] not in valid or ident in seen:raise ValueError('bad candidate instance ID')
            seen.add(ident);counts[match[1]].append(int(match[2]))
            runs=string_to_counts(row['segmentation_rle'])
            if sum(runs)!=2048*2048 or any(v<0 for v in runs) or sum(runs[1::2])<=0:
                raise ValueError('bad COCO mask')
            to_decode.append(row['segmentation_rle'])
    for k,items in counts.items():
        if items!=list(range(1,len(items)+1)):raise ValueError('nonconsecutive IDs '+k)
    rng=random.Random(2026)
    for index in rng.sample(range(len(to_decode)),min(12,len(to_decode))):
        assert decode(to_decode[index],(2048,2048)).any()
    assert len(seen)==meta['predictions']
    checksum=hashlib.sha256(p.read_bytes()).hexdigest();assert checksum==meta['sha256']
    answer={'verified_rows':len(seen),'images_covered':180,'images_without_filaments':sum(not x for x in counts.values()),
            'random_2048_decoded':min(12,len(to_decode)), 'all_rle_lengths_checked':True,
            'file_sha256':checksum, 'no_test_labels_read':True}
    (RESULT/'SOLAR_SUBMISSION_CSV_QA.json').write_text(json.dumps(answer,indent=2))
    return answer

if __name__=='__main__':print(json.dumps(verify(),indent=2))
