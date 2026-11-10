from pathlib import Path
import csv,json
from solar_filament.coco_rle import encode,decode
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_rle_coco_roundtrip():
    x=np.zeros((65,37),np.uint8);x[4:55,8:11]=1
    assert np.array_equal(decode(encode(x),x.shape),x)


def test_checkpoint_present():
    import torch
    d=torch.load(ROOT/'results/SOLAR_REAL_TRAINED_UNET.pt',map_location='cpu',weights_only=True)
    assert d['base']==8 and 'state_dict' in d


def test_real_source_metadata_present():
    s=json.loads((ROOT/'results/SOLAR_DATA_AUDIT.json').read_text())
    assert s['total_images']==707 and s['annotator_views']==1154


def test_edge_candidate_header():
    with (ROOT/'results/SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv').open() as f:
        assert next(csv.reader(f))==['filament_id','segmentation_rle']
