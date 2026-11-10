"""Atomic, resumable, complete 180-image Kaggle prediction reproduction.

This program does NOT train, use hidden labels, seed from previous predictions,
or require the earlier competition outputs. Every missing shard is actually
inferred using the checked-in PyTorch weights and official test JPEG only.

A generated shard is committed only after its SHA-256, source JPEG CRC and
model SHA-256 are saved. The final CSV is committed only after all 180 images
are validated and the completed candidate is independently decoded.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,zipfile
from pathlib import Path
import cv2
import numpy as np
from solar_real import ROOT,RESULT,read_model,Z2
from solar_tta import predict_tta
from solar_eval import instance_map
from refined_inference import edge_refine,sparse_rle
from resumable_inference import load_verified,atomic_save,sha
from solar_source import OfficialTestImages
from verify_candidate import verify

DEFAULT_TEST=RESULT/'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv'
DEFAULT_MODEL=RESULT/'SOLAR_REAL_TRAINED_UNET.pt'


def _atomic_text(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'w',encoding='utf-8',newline='') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def reproduce(source_zip=Z2,output=None,shard_dir=None,reference=None,first=None,count=None):
    """Generate exact official CSV from original unlabelled test images.

    first/count can be used for small smoke tests; full output requires 180.
    """
    source_zip=Path(source_zip)
    output=Path(output or RESULT/'SOLAR_REGENERATED_180.csv')
    shard_dir=Path(shard_dir or ROOT/'_repro_shards')
    reference=Path(reference) if reference else None
    modelsha=sha(DEFAULT_MODEL)
    with OfficialTestImages(source_zip) as src:
        paths=src.paths
    items=list(paths.items())
    if first is not None:
        if count is None:raise ValueError('count required for subset reproduction')
        items=items[first:first+count]
    if not items:raise ValueError('No selected official test photos')
    n_new=0;n_reused=0
    model=None
    with OfficialTestImages(source_zip) as source:
        for n,(stem,location) in enumerate(items,1):
            crc=source.crc(stem)
            try:
                rows=load_verified(shard_dir,stem,crc,modelsha)
                n_reused+=1
            except (FileNotFoundError,KeyError,ValueError,TypeError) as error:
                # Do not import checkpoint until the first real inference.
                if model is None:model=read_model()
                b=source.bytes(stem)
                image=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape!=(2048,2048):raise ValueError(f'Invalid official test JPEG {location}')
                small=cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA)
                probability=predict_tta(model,small)
                config=json.loads((RESULT/'SOLAR_TTA_COMPARISON.json').read_text())['parameters']
                instance_map512=instance_map(probability,small,**config)
                full=edge_refine(instance_map512,probability,threshold=.45)
                rows=[]
                for j in range(1,int(instance_map512.max())+1):
                    encoded=sparse_rle(full,j)
                    if encoded is not None:rows.append((stem+'_'+str(len(rows)+1),encoded))
                atomic_save(shard_dir,stem,rows,crc,modelsha,'actual_inference_official_test_JPEG')
                # Read data back to prevent unverified shard state.
                check=load_verified(shard_dir,stem,crc,modelsha)
                if [list(row) for row in rows]!=check:raise ValueError(f'Shard read-back mismatch {stem}')
                n_new+=1
            if n%30==0 or n==len(items):
                print(f'REPRODUCE {n}/{len(items)}; freshly_inferred={n_new}, verified_reused={n_reused}',flush=True)
    summary={'selected_images':len(items),'newly_inferred':n_new,'existing_sha_verified_shards':n_reused,
             'official_test_JPEG_zip':source_zip.name,'trained_checkpoint_sha256':modelsha,
             'test_ground_truth_used':False,'reference_csv':reference.name if reference else None,
             'output_csv':None,'same_as_reference':None}
    # A partial shard run may NOT be mislabeled as a complete submission.
    if len(items)!=180:
        return summary
    temp=output.with_name(output.name+'.tmp')
    temp.parent.mkdir(parents=True,exist_ok=True)
    with OfficialTestImages(source_zip) as source,open(temp,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for stem,location in paths.items():
            writer.writerows(load_verified(shard_dir,stem,source.crc(stem),modelsha))
        f.flush();os.fsync(f.fileno())
    audit=verify(temp,source_zip,decode_count=30)
    if audit['images']!=180 or audit['rows']==0:raise ValueError('Incomplete final CSV')
    produced=sha(temp)
    if reference is not None:
        matching=produced==sha(reference)
        summary['same_as_reference']=matching
        if not matching:
            raise ValueError(f'Full regeneration differs from saved official candidate; kept {temp} for investigation')
    os.replace(temp,output)
    summary.update({'output_csv':str(output),'csv_rows':audit['rows'],'candidate_sha256':produced,
                    'images_without_predictions':audit['empty_images'],'real_2048_masks_decoded':audit['decoded_masks'],
                    'atomic_commit':True})
    _atomic_text(output.with_suffix('.audit.json'),json.dumps(summary,indent=2)+'\n')
    print('REPRODUCTION COMPLETE:',json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--test-zip',default=str(Z2));p.add_argument('--output')
    p.add_argument('--shards');p.add_argument('--reference',default=str(DEFAULT_TEST))
    p.add_argument('--first',type=int);p.add_argument('--count',type=int)
    args=p.parse_args()
    reproduce(args.test_zip,args.output,args.shards,args.reference,args.first,args.count)
