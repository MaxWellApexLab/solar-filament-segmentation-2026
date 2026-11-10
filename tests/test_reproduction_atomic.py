import json
from pathlib import Path
import pytest
from solar_real import ROOT,RESULT
from reproduce_submission import reproduce


def test_saved_reference_output_present():
    p=RESULT/'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv'
    assert p.is_file() and p.stat().st_size>200000


def test_real_2_image_fresh_recompute(tmp_path,monkeypatch):
    """Genuine model inference, original Kaggle JPEGs, no seeded predictions."""
    from solar_real import Z2
    if not Z2.exists():pytest.skip('Authorized Kaggle test ZIP is not mounted')
    r=reproduce(source_zip=Z2,shard_dir=tmp_path/'shards',reference=None,first=0,count=2)
    assert r['newly_inferred']==2 and r['selected_images']==2
    assert len(list((tmp_path/'shards').glob('*.csv.meta.json')))==2
    again=reproduce(source_zip=Z2,shard_dir=tmp_path/'shards',reference=None,first=0,count=2)
    assert again['newly_inferred']==0 and again['existing_sha_verified_shards']==2
