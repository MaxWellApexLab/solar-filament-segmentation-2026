from pathlib import Path
import hashlib,json,os
import nbformat,pytest
from pypdf import PdfReader
from solar_source import OfficialTestImages
from solar_real import ROOT,Z2


def test_native_real_measurement_is_preserved():
    r=json.loads((ROOT/'results/SOLAR_ORIGINAL2048_EDGE_REPRODUCED.json').read_text())
    assert (r['n_images'],r['n_rater_views'])==(62,93)
    assert r['edge_refined_TTA']['pq']>r['baseline_TTA']['pq']
    assert r['bootstrap_95_percentile_CI'][0]>0


def test_notebook_14_cells_completed_without_errors():
    p=ROOT/'notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb'
    if not p.exists():pytest.skip('Executed artifact not built')
    nb=nbformat.read(p,as_version=4)
    assert len(nb.cells)==14
    for c in nb.cells:
        if c.cell_type=='code':
            assert isinstance(c.get('execution_count'),int)
            assert not any(o.output_type=='error' for o in c.outputs)


def test_full_report_is_four_pages():
    doc=PdfReader(str(ROOT/'report/technical_report_4pages.pdf'))
    assert len(doc.pages)==4
    content=' '.join(p.extract_text() or '' for p in doc.pages)
    assert '0.2731' in content and '1,013' in content


def test_Kaggle_directory_mode_crc_parity():
    direct=ROOT/'_kaggle_input_data'
    if not direct.exists() or not Z2.exists():pytest.skip('Organizer datasets not attached')
    with OfficialTestImages(direct) as a,OfficialTestImages(Z2) as b:
        assert a.paths.keys()==b.paths.keys()
        assert all(a.crc(stem)==b.crc(stem) for stem in a.paths)


def test_explicit_organizer_id_format_and_sha():
    p=ROOT/'submission.csv'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='e0e55cf4eb383769c17b742d556d619859307b82a4267792ad2345698126d9ea'
    assert p.read_text().splitlines()[0]=='filament_id,segmentation_rle'
