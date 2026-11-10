"""Solar-only model and COCO RLE smoke tests (no ClimX dependency)."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
from solar_real import ROOT,RESULT,CACHE
from solar_eval import upscale_rle
from solar_filament.coco_rle import encode,decode


def test_date_separation():
    # Original Kaggle JPEG and polygons are deliberately not redistributed in
    # this public source repository. Audited partition counts are preserved.
    r=json.loads((RESULT/'SOLAR_DATA_AUDIT.json').read_text())
    assert r['total_images']==707 and r['annotator_views']==1154
    assert r['splits']=={'train':582,'tune':63,'holdout':62}
    assert r['test_used'] is False


def test_exact_sparse_rle():
    rng=np.random.default_rng(73)
    for size in [(6,7),(8,3),(19,25)]:
        for _ in range(13):
            m=(rng.random(size)>.72)
            expanded=np.repeat(np.repeat(m,4,axis=0),4,axis=1).astype(np.uint8)
            e=upscale_rle(m)
            assert e==encode(expanded)
            assert np.array_equal(decode(e,expanded.shape),expanded)


def test_solar_checkpoint():
    import torch
    ck=torch.load(RESULT/'SOLAR_REAL_TRAINED_UNET.pt',weights_only=True,map_location='cpu')
    assert ck['train_size']==582 and ck['base']==8 and ck['epoch']>=1


def test_native_polygon_rasterization():
    from solar_fullres_check import rasterize_original
    ann=[{'segmentation':[[100,100,150,100,150,150,100,150]]}]
    r=rasterize_original(ann)
    assert r.shape==(2048,2048) and r.dtype==np.int16
    assert r[125,125]==1 and r[50,50]==0


def test_instance_dilation_keeps_distinct_ids():
    from solar_fullres_refinement import enlarge
    m=np.zeros((15,15),np.int16);m[5,4]=1;m[5,10]=2
    out=enlarge(m,2)
    assert out[5,4]==1 and out[5,10]==2
    assert out.max()==2 and np.count_nonzero(out)>np.count_nonzero(m)


def test_testtime_augmentation_inverse_flips():
    from solar_tta import predict_tta
    import torch
    model=torch.nn.Identity()
    img=np.arange(48,dtype=np.uint8).reshape(6,8)
    q=predict_tta(model,img)
    assert q.shape==img.shape
    assert np.allclose(q,torch.sigmoid(torch.from_numpy(img.astype(np.float32)/255.)).numpy(),atol=1e-7)
