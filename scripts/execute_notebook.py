"""Reproduce all competition test predictions from the executed Jupyter pipeline.

An execution error or missing official test images is fatal, not silently skipped.
"""
import os,sys,nbformat,tempfile
from pathlib import Path
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
nb=nbformat.read(ROOT/'notebooks/REPRODUCE_SOLAR.ipynb',as_version=4)
print('RUN NOTEBOOK in',ROOT,flush=True)
try:
    out=NotebookClient(nb,kernel_name='python3',timeout=300,allow_errors=False,resources={'metadata':{'path':str(ROOT)}}).execute()
    assert all(c.get('execution_count') is not None for c in out.cells if c.cell_type=='code')
    dest=ROOT/'notebooks/REPRODUCE_SOLAR_EXECUTED.ipynb'
    tmp=dest.with_name(dest.name+'.tmp')
    nbformat.write(out,tmp)
    os.replace(tmp,dest)
    print('NOTEBOOK SUCCESS',len(out.cells),'cells, output',dest,flush=True)
except Exception:
    print('NOTEBOOK FAILED; existing prior executed notebook left unchanged',flush=True)
    raise
