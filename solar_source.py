"""Official MAGFiLO 2026 test JPEG reader: zip archive OR extracted Kaggle dataset.

Does not load labels or metadata besides the filenames and JPEG bytes. The same
PNG/JPEG byte CRC is used in resumable inference regardless of container type.
"""
from pathlib import Path
import zlib,zipfile

class OfficialTestImages:
    def __init__(self,root):
        self.root=Path(root)
        if not self.root.exists():raise FileNotFoundError(self.root)
        self.z=None
        if self.root.is_file():
            if not zipfile.is_zipfile(self.root):raise ValueError('File is not a supported competition ZIP')
            self.z=zipfile.ZipFile(self.root)
            entries=sorted(x for x in self.z.namelist() if '/test/test_images/' in x and x.lower().endswith('.jpeg'))
            self._paths={Path(x).stem:x for x in entries}
        else:
            entries=sorted(self.root.rglob('*.jpeg'))
            entries=[x for x in entries if x.parent.name=='test_images' and x.parent.parent.name=='test']
            self._paths={x.stem:x for x in entries}
        if len(entries)!=180 or len(self._paths)!=180:
            self.close()
            raise ValueError('Exactly 180 distinct official test JPEGs are required')
    def __enter__(self):return self
    def __exit__(self,exctype,excvalue,tb):self.close()
    def close(self):
        if self.z is not None:
            self.z.close();self.z=None
    @property
    def paths(self):return dict(sorted(self._paths.items()))
    def bytes(self,stem):
        p=self._paths[stem]
        return self.z.read(p) if self.z is not None else p.read_bytes()
    def crc(self,stem):
        p=self._paths[stem]
        return self.z.getinfo(p).CRC if self.z is not None else zlib.crc32(p.read_bytes())&0xffffffff
