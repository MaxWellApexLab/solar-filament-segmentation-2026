"""Strict independent release checks; emits atomic machine-readable receipt."""
import csv,hashlib,json,os,sys
from pathlib import Path
import nbformat,torch
from pypdf import PdfReader
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solar_source import OfficialTestImages
from solar_filament.coco_rle import string_to_counts
from verify_candidate import verify


def digest(path):
    sha=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):sha.update(part)
    return sha.hexdigest()


def run(source=None):
    R=ROOT;result=R/'results';submission=R/'submission.csv';report=R/'report/technical_report_4pages.pdf'
    assert submission.is_file() and report.is_file()
    checkpoint=result/'SOLAR_REAL_TRAINED_UNET.pt'
    data=torch.load(checkpoint,weights_only=True,map_location='cpu')
    assert data['base']==8 and data['train_size']==582 and data['epoch']>=1
    model_sha=digest(checkpoint)
    assert model_sha=='a0183861865e2cb3e152b8640bcc61c503d66b206c9fcfc9ed682a6c7fe6349f'
    expected='e0e55cf4eb383769c17b742d556d619859307b82a4267792ad2345698126d9ea'
    assert digest(submission)==expected
    assert digest(result/'SOLAR_MAGFiLO_2026_TTA_NATIVE_EDGE_180_CANDIDATE.csv')==expected
    assert digest(result/'SOLAR_REGENERATED_180_VERIFIED.csv')==expected
    # Whole-file RLE check independent of the hidden labels.
    ids=set();by={};rows=0
    with submission.open(newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['filament_id','segmentation_rle']
        for obj in reader:
            name=obj['filament_id'];stem,_,num=name.rpartition('_')
            assert stem and num.isdigit() and int(num)>0 and name not in ids
            ids.add(name);by.setdefault(stem,[]).append(int(num))
            counts=string_to_counts(obj['segmentation_rle'])
            assert sum(counts)==2048*2048 and sum(counts[1::2])>0 and all(x>=0 for x in counts)
            rows+=1
    assert rows==1013 and len(by)==174
    assert all(ids==list(range(1,len(ids)+1)) for ids in by.values())
    # Original test directory or authorized ZIP is present for a full ID audit.
    test_source=Path(source) if source else Path(os.environ.get('SOLAR_ZIP2','/nonexistent'))
    if test_source.exists():
        reportdata=verify(submission,test_source,decode_count=30)
        assert reportdata['images']==180 and reportdata['rows']==1013
        assert reportdata['sha256']==expected
    q=PdfReader(str(report));assert len(q.pages)==4
    combined=' '.join(p.extract_text() or '' for p in q.pages)
    assert '0.2731' in combined and '0.2629' in combined and '1,013' in combined
    assert '\\documentclass[sigconf,nonacm]{acmart}' in (R/'report/main.tex').read_text()
    assert 'MIT License' in (R/'LICENSE').read_text()
    # Executed notebook must show every code cell genuinely ran.
    nb=nbformat.read(R/'notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb',as_version=4)
    assert len(nb.cells)>=14
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(isinstance(c.get('execution_count'),int) for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    receipt=json.loads((result/'SOLAR_NOTEBOOK_EXECUTION_RECEIPT.json').read_text())
    assert receipt['csv_sha256']==expected and receipt['equals_published_candidate'] and receipt['verified_real_test_images']==180
    assert not receipt['formal_submission_completed']
    reqs=[Requirement(x.strip()) for x in (R/'requirements.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
    assert len(reqs)>=10 and all('==' in str(r.specifier) for r in reqs)
    # Every source-code dependency version is pinned and the original check was recorded.
    audit=json.loads((result/'SOLAR_ORIGINAL2048_EDGE_REPRODUCED.json').read_text())
    assert audit['n_images']==62 and audit['n_rater_views']==93 and abs(audit['edge_refined_TTA']['pq']-.27311477742304735)<1e-10
    status={'result':'PRE_PUBLICATION_ENGINEERING_VERIFIED',
      'official_submission_accepted':False,'official_leaderboard_score':None,
      'participant_actions_required':['Set real author and affiliation in ACM report, approve open-source/data/model terms','Publish source repository','Upload plain CSV to Kaggle','Submit final organizer Google Form'],
      'trained_checkpoint_sha256':model_sha,'submission_sha256':expected,'all_180_real_test_JPEG_inferred_from_weights':True,
      'source_input_modes_verified':['authorized_original_ZIP','extracted_kaggle_test_directory'],
      'prediction_rows':rows,'test_images':180,'empty_images':6,'original_polygon_audit_images':62,
      'original_polygon_audit_views':93,'local_native_PQ':audit['edge_refined_TTA']['pq'],
      'notebook_code_cells_executed':len(code),'pdf_pages':len(q.pages),'pinned_requirements':len(reqs),
      'source_zip_used_for_full_ID_audit':test_source.name if test_source.exists() else None,
      'computed_without_any_hidden_test_labels':True}
    folder=R/'release';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'FINAL_ENGINEERING_VERIFICATION_RECEIPT.json';temp=dest.with_name(dest.name+'.tmp')
    with temp.open('w') as f:
        json.dump(status,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,dest)
    print('FINAL PREFLIGHT SUCCESS',json.dumps(status,indent=2),flush=True)
    return status

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else None)
