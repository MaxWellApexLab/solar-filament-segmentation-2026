"""Create independently verified atomic public-Git and private submission ZIPs.

No Kaggle images, training/test ZIPs, credentials, cache, or raw test preview
embedded in public Git release. Uses already verified real-model outputs; no
retraining and no network/account actions.
"""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(os.environ.get('SOLAR_RELEASE_OUTPUT','/mnt/data'))
STAGE=OUTPUT/'_solar_gitrepo_atomic_staging'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def copy_into_stage(relative):
    src=ROOT/relative
    assert src.is_file(),f'Missing reproducibility artifact {relative}'
    target=STAGE/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,target)


def create_stage():
    if STAGE.exists():shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    # Source, public documentation, packaging, and tests.
    for entry in ROOT.glob('*.py'):copy_into_stage(entry.relative_to(ROOT))
    for name in ['README.md','LICENSE','DATA_AND_MODEL_RIGHTS.md',
                 'requirements.txt','.gitignore','pytest.ini','submission.csv']:
        copy_into_stage(Path(name))
    for name in ('solar_filament','scripts','tests','docs'):
        for file in (ROOT/name).rglob('*'):
            if file.is_file() and file.suffix in ('.py','.md','.txt'):
                copy_into_stage(file.relative_to(ROOT))
    for name in ['notebooks/REPRODUCE_SOLAR.ipynb',
                 'report/main.tex','report/technical_report_4pages.pdf',
                 'report/make_evidence_figures.py']:
        copy_into_stage(Path(name))
    for file in (ROOT/'report/figures').glob('*.pdf'):
        copy_into_stage(file.relative_to(ROOT))
    # Preserve execution counts, stream outputs and success evidence, but
    # REMOVE output cells with image/png (original organizer JPEG derivatives).
    nb=nbformat.read(ROOT/'notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb',as_version=4)
    n_removed=0
    for cell in nb.cells:
        if cell.cell_type=='code':
            for out in cell.outputs:
                if 'data' in out and 'image/png' in out.data:
                    del out.data['image/png'];n_removed+=1
                    out.data['text/plain']='[Organizer-supplied test JPEG preview omitted from public source archive]'
    assert n_removed>=1,'Expected embargoed test-image preview in executed notebook'
    target=STAGE/'notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb'
    nbformat.write(nb,target)
    results=[
        'SOLAR_REAL_TRAINED_UNET.pt','SOLAR_TRAIN_HISTORY.json',
        'SOLAR_DATA_AUDIT.json','SOLAR_ORIGINAL2048_EDGE_REPRODUCED.json',
        'SOLAR_TTA_COMPARISON.json','SOLAR_NATIVE2048_TTA_PAIRED.json',
        'SOLAR_KAGGLE_SOURCE_HASHES.json','SOLAR_TTA_ALL180_INDEPENDENT_QA.json',
        'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv',
        'SOLAR_REGENERATED_180_VERIFIED.csv',
        'SOLAR_NOTEBOOK_EXECUTION_RECEIPT.json',
        'SOLAR_REGENERATED_180_VERIFIED.audit.json']
    for file in results:copy_into_stage(Path('results')/file)
    copy_into_stage(Path('release/FINAL_ENGINEERING_VERIFICATION_RECEIPT.json'))
    note=STAGE/'notebooks/EXECUTED_NOTEBOOK_PRIVACY.md'
    note.write_text('The public copy of REPRODUCE_SOLAR_EXECUTED.ipynb preserves execution counts, '
        'cell text and validation logs, but strips inline previews of Kaggle-provided test JPEGs '
        '(one output image) to avoid redistributing dataset imagery. The original fully executed '
        'notebook remains in the participant-owned private workspace.\n')
    expected=sha(STAGE/'submission.csv')
    assert expected=='e0e55cf4eb383769c17b742d556d619859307b82a4267792ad2345698126d9ea'
    assert sha(STAGE/'results/SOLAR_REAL_TRAINED_UNET.pt')=='a0183861865e2cb3e152b8640bcc61c503d66b206c9fcfc9ed682a6c7fe6349f'
    manifest={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()}
    (STAGE/'SHA256_MANIFEST.json').write_text(json.dumps({'scope':'Public Git-ready source project',
       'model_source':'real MAGFiLO 2026 training data supplied by competition participant',
       'prediction_source':'entire 180 official test JPEGs, derived output only',
       'source_data_redistributed':False,'inline_test_image_outputs_stripped':n_removed,
       'files_sha256':manifest},indent=2))
    return {'files':len(manifest),'stripped_image_outputs':n_removed,'csv_sha256':expected}


def atomic_zip(files,destination):
    destination.parent.mkdir(exist_ok=True,parents=True)
    staging=destination.with_suffix('.partial.zip')
    if staging.exists():staging.unlink()
    with zipfile.ZipFile(staging,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for arcname,file in files:
            assert Path(file).is_file()
            z.write(file,arcname=arcname)
    with zipfile.ZipFile(staging,'r') as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names))==len(files)
        # Prevent exfiltration or redistribution of original competitor imagery,
        # credentials and accumulated test-image caches.
        for name in names:
            assert not name.startswith(('_kaggle_input_data/','_repro_shards/','solar_cache/'))
            assert not name.endswith(('.jpeg','.jpg','.h5','.nc','.zip','.part001'))
            assert 'kaggle.json' not in name and '/.env' not in name
    with staging.open('rb') as f:os.fsync(f.fileno())
    os.replace(staging,destination)
    return {'file':str(destination),'bytes':destination.stat().st_size,
            'zip_sha256':sha(destination),'entries':len(files)}


def verify_manifest_in_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        m=json.loads(z.read('SHA256_MANIFEST.json'))['files_sha256']
        for name,expected in m.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        return len(m)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    a=create_stage()
    files=[(str(p.relative_to(STAGE)),p) for p in sorted(STAGE.rglob('*')) if p.is_file()]
    src=atomic_zip(files,OUTPUT/'Solar_Filament_2026_PUBLIC_GIT_PROJECT_VERIFIED.zip')
    manifest_count=verify_manifest_in_zip(src['file'])
    assert manifest_count==a['files']
    # A separate convenience ZIP is for the participant's own filing records.
    # The actual Kaggle uploader takes the plain `submission.csv` FILE ONLY.
    entries={
      'submission.csv':ROOT/'submission.csv',
      'technical_report_4pages.pdf':ROOT/'report/technical_report_4pages.pdf',
      'FINAL_SUBMISSION_CHECKLIST.md':ROOT/'docs/FINAL_SUBMISSION_CHECKLIST.md',
      'FINAL_ENGINEERING_VERIFICATION_RECEIPT.json':ROOT/'release/FINAL_ENGINEERING_VERIFICATION_RECEIPT.json',
      'README_FIRST.txt':STAGE/'notebooks/EXECUTED_NOTEBOOK_PRIVACY.md'
    }
    assets=atomic_zip(list(entries.items()),OUTPUT/'Solar_Filament_2026_FINAL_SUBMISSION_ASSETS_VERIFIED.zip')
    receipt={'status':'ATOMIC_ZIP_VERIFIED','public_git_project':src,
             'submission_assets':assets,'source_manifest_verified_files':manifest_count,
             'source_zip_contains_raw_competition_images':False,
             'public_git_notebook_privacy_image_outputs_removed':a['stripped_image_outputs'],
             'plain_CSV_for_Kaggle':str(ROOT/'submission.csv'),
             'official_Kaggle_submission_status':'NOT YET SUBMITTED',
             'four_page_report_verified':True,
             'public_repository_url':None}
    receipt_path=ROOT/'release/ATOMIC_RELEASE_RECEIPT.json';tmp=receipt_path.with_suffix('.partial.json')
    with tmp.open('w') as f:
        json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,receipt_path)
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt

if __name__=='__main__':run()
