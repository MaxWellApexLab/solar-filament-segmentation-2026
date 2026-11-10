"""Safe, resumable Solar Kaggle candidate inference without redundant retraining.

One per-image shard is complete only after SHA-256 metadata is atomically
written. The previously verified v5 candidate can seed shards, avoiding 180
identical repeated inferences. `recompute` independently reruns any chosen
image from the SAVED trained checkpoint and byte-compares to its seeded shard.
`assemble` requires 180 explicitly verified shards, including zero-instance
images, and checks the full candidate SHA-256 against the v5 original.
No Kaggle hidden labels, no credentials, no leaderboard claims.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2,torch
from solar_real import RESULT,Z2,read_model
from solar_tta import predict_tta
from solar_eval import instance_map
from refined_inference import edge_refine,sparse_rle
from solar_filament.coco_rle import string_to_counts

DEFAULT=RESULT/'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def official_paths(sourcezip):
    with zipfile.ZipFile(sourcezip) as z:
        names=sorted(n for n in z.namelist() if '/test/test_images/' in n and n.endswith('.jpeg'))
        if len(names)!=180:raise ValueError(f'Expected 180 official test JPEGs, found {len(names)}')
        stems=[Path(n).stem for n in names]
        if len(set(stems))!=180:raise ValueError('Duplicate test photo IDs')
        return dict(zip(stems,names))


def shardfile(shard_dir,stem):
    return Path(shard_dir)/(stem+'.csv')


def _encode_rows(rows):
    stream=io.StringIO(newline='')
    w=csv.writer(stream)
    for row in rows:w.writerow(row)
    return stream.getvalue().encode('utf-8')


def atomic_save(shard_dir,stem,rows,source_crc,checkpoint_hash,origin):
    base=Path(shard_dir);base.mkdir(parents=True,exist_ok=True)
    # Never trust arbitrary names for filesystem paths. Photo IDs must be
    # official fixed-format basenames and contain no path traversal.
    if not stem or not all(x.isalnum() or x in '_-' for x in stem):raise ValueError('Malformed image ID')
    raw=_encode_rows(rows)
    target=shardfile(base,stem)
    meta={'image_id':stem,'image_crc32':int(source_crc),'rows':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),
          'checkpoint_sha256':checkpoint_hash,'source':origin,'valid':True}
    tmp=target.with_suffix('.csv.tmp');tmp.write_bytes(raw);os.replace(tmp,target)
    tmp_meta=Path(str(target)+'.meta.tmp');tmp_meta.write_text(json.dumps(meta,sort_keys=True));os.replace(tmp_meta,str(target)+'.meta.json')
    return meta


def load_verified(shard_dir,stem,expected_crc,expected_checkpoint_sha):
    p=shardfile(shard_dir,stem)
    meta_path=Path(str(p)+'.meta.json')
    if not (p.exists() and meta_path.exists()):raise FileNotFoundError(f'Missing complete shard for {stem}')
    meta=json.loads(meta_path.read_text())
    if not meta.get('valid') or meta.get('image_id')!=stem or int(meta['image_crc32'])!=int(expected_crc) or meta['checkpoint_sha256']!=expected_checkpoint_sha:
        raise ValueError('Shard provenance mismatch for '+stem)
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Corrupted shard '+stem)
    rows=list(csv.reader(io.StringIO(raw.decode('utf-8'))))
    if len(rows)!=meta['rows']:raise ValueError('Incorrect row count '+stem)
    for i,(ident,encoded) in enumerate(rows,1):
        if ident!=stem+'_'+str(i):raise ValueError('Unordered image object ID '+ident)
        runs=string_to_counts(encoded)
        if sum(runs)!=2048*2048 or any(x<0 for x in runs) or sum(runs[1::2])==0:
            raise ValueError('Invalid COCO RLE '+ident)
    return rows


def seed_from_saved(saved_candidate,sourcezip,checkpoint,shard_dir):
    source=official_paths(sourcezip);modelsha=sha(checkpoint)
    by_image=defaultdict(list)
    with open(saved_candidate,newline='') as f:
        rdr=csv.reader(f)
        if next(rdr)!=['filament_id','segmentation_rle']:raise ValueError('Wrong saved CSV header')
        for ident,rle in rdr:
            stem,sep,num=ident.rpartition('_')
            if not sep or stem not in source or not num.isdigit():raise ValueError('Bad official test ID '+ident)
            by_image[stem].append((ident,rle))
    with zipfile.ZipFile(sourcezip) as z:
        for stem,path in source.items():
            rows=by_image.get(stem,[])
            if [r[0] for r in rows]!=[f'{stem}_{i}' for i in range(1,len(rows)+1)]:raise ValueError('Non-consecutive instance IDs')
            atomic_save(shard_dir,stem,rows,z.getinfo(path).CRC,modelsha,'v5_vetted_saved_candidate')
    return {'complete_shards':180,'rows':sum(len(x) for x in by_image.values()),
            'checkpoint_sha256':modelsha,'source_csv_sha256':sha(saved_candidate),
            'note':'SHARDING EXISTING VERIFIED PREDICTIONS, NOT 180 NEW INFERENCE RUNS'}


def assemble(sourcezip,checkpoint,shard_dir,output,expected_sha=None):
    paths=official_paths(sourcezip);modelsha=sha(checkpoint);allrows=[]
    with zipfile.ZipFile(sourcezip) as z:
        for stem,path in paths.items():
            allrows.extend(load_verified(shard_dir,stem,z.getinfo(path).CRC,modelsha))
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows(allrows)
    result={'test_images':180,'instance_rows':len(allrows),'sha256':sha(p),
            'matches_saved_candidate':None if expected_sha is None else sha(p)==expected_sha}
    if expected_sha is not None and not result['matches_saved_candidate']:raise ValueError('Assembled candidate differs from authoritative saved CSV')
    return result


def recompute_sample(sourcezip,checkpoint,shard_dir,first=0,n=5,compare_only=True):
    paths=official_paths(sourcezip);selected=list(paths.items())[first:first+n]
    if not selected:raise ValueError('Empty image range')
    modelsha=sha(checkpoint);config=json.loads((RESULT/'SOLAR_TTA_COMPARISON.json').read_text())['parameters']
    torch.set_num_threads(3);model=read_model()
    result=[]
    with zipfile.ZipFile(sourcezip) as z:
        for stem,path in selected:
            img=cv2.imdecode(np.frombuffer(z.read(path),np.uint8),cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape!=(2048,2048):raise ValueError('Bad official test frame')
            sample=cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
            proba=predict_tta(model,sample)
            labels=instance_map(proba,sample,**config)
            refined=edge_refine(labels,proba)
            rows=[]
            for item in range(1,int(labels.max())+1):
                rle=sparse_rle(refined,item)
                if rle is not None:rows.append((f'{stem}_{len(rows)+1}',rle))
            prior=load_verified(shard_dir,stem,z.getinfo(path).CRC,modelsha)
            equal=[list(x) for x in rows]==prior
            if compare_only and not equal:raise ValueError('Reconstructed model output differs from saved candidate '+stem)
            if not compare_only:atomic_save(shard_dir,stem,rows,z.getinfo(path).CRC,modelsha,'recomputed_v5_model')
            result.append({'image':stem,'count':len(rows),'exact_match_to_saved':equal})
            print('INFER_REPLAY',stem,'count',len(rows),'exact=',equal,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['seed','assemble','recompute'])
    parser.add_argument('--zip',default=str(Z2));parser.add_argument('--checkpoint',default=str(RESULT/'SOLAR_REAL_TRAINED_UNET.pt'))
    parser.add_argument('--saved',default=str(DEFAULT));parser.add_argument('--shards',required=True)
    parser.add_argument('--output');parser.add_argument('--first',type=int,default=0);parser.add_argument('--count',type=int,default=5)
    a=parser.parse_args()
    if a.command=='seed':out=seed_from_saved(a.saved,a.zip,a.checkpoint,a.shards)
    elif a.command=='assemble':out=assemble(a.zip,a.checkpoint,a.shards,a.output or DEFAULT+'.reassembled.csv',sha(a.saved))
    else:out=recompute_sample(a.zip,a.checkpoint,a.shards,a.first,a.count)
    print(json.dumps(out,indent=2))
