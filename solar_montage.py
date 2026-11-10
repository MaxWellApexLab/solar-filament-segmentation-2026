"""Real official training image overlays for scientific error diagnostics."""
import json
from PIL import Image,ImageDraw,ImageFont
import cv2,numpy as np
from solar_real import RESULT,CACHE,read_model,predict
from solar_eval import split,instance_map


def generate():
    report=json.loads((RESULT/'SOLAR_REAL_LOCAL_PQ_REPORT.json').read_text())
    cfg=report['selection']['config']
    rows=sorted(split()['tune'],key=lambda r:r['name'])[::max(1,len(split()['tune'])//6)][:6]
    net=read_model()
    out=Image.new('RGB',(1536,1100),'#0b1524');dr=ImageDraw.Draw(out)
    for i,r in enumerate(rows):
        with np.load(CACHE/r['cache_file']) as z:
            im=z['im'];gt=z['inst'][0]>0
        pred=instance_map(predict(net,im),im,**cfg)>0
        gray=np.stack((im,im,im),axis=2).astype(float)
        # Pred-only = red, GT-only = green, overlap = teal
        onlyp=pred&~gt;onlyg=gt&~pred;both=pred&gt
        gray[onlyp]=.5*gray[onlyp]+np.array([255,60,70])*.5
        gray[onlyg]=.5*gray[onlyg]+np.array([70,240,100])*.5
        gray[both]=.4*gray[both]+np.array([60,230,220])*.6
        small=Image.fromarray(np.uint8(np.clip(gray,0,255)))
        x=(i%3)*512;y=(i//3)*550
        out.paste(small,(x,y+28))
        dr.text((x+7,y+8),r['name'][:18],fill='white')
    dest=RESULT/'SOLAR_REAL_GONG_TUNE_ERROR_OVERLAY.png'
    out.save(dest)
    print(dest)
    return dest

if __name__=='__main__':generate()
