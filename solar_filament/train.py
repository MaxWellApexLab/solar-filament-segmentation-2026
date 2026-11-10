"""Train official-data-only U-Net; produces own weights/metadata (not pretrained)."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from .dataset import CropDataset,group_coco,date_holdout
from .train_unet import FilamentUNet,combined_loss


def train(image_root,annotations,output,epochs=2,batch_size=4,device='cpu',seed=2026):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    groups=group_coco(annotations);train_names,val_names=date_holdout(groups,seed=seed)
    if not train_names or not val_names:raise ValueError('need >=2 observation dates for leakage-free split')
    model=FilamentUNet(base=16).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    train_loader=DataLoader(CropDataset(image_root,annotations,train_names,patches_per_image=3,seed=seed),batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(CropDataset(image_root,annotations,val_names,patches_per_image=1,seed=seed+50000),batch_size=batch_size,shuffle=False,num_workers=0)
    history=[];best=float('inf')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train();train_values=[]
        for x,y in train_loader:
            x=x.to(device);y=y.to(device);opt.zero_grad();loss=combined_loss(model(x),y);loss.backward();opt.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval();v=[]
        with torch.no_grad():
            for x,y in val_loader:v.append(float(combined_loss(model(x.to(device)),y.to(device)).cpu()))
        row={'epoch':epoch+1,'train_loss':float(np.mean(train_values)),'val_loss':float(np.mean(v))}
        history.append(row)
        if row['val_loss']<best:
            best=row['val_loss']
            torch.save({'state_dict':model.state_dict(),'seed':seed,'train_images':len(train_names),
                        'val_images':len(val_names),'epoch':epoch+1,'base':16},out)
    (out.parent/(out.stem+'_history.json')).write_text(json.dumps({'history':history,'date_holdout':True,'training_data':'organizer MAGFiLO train only','public_score':None},indent=2))
    return history

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image-root',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--device',default='cpu')
    a=p.parse_args();print(train(a.image_root,a.annotations,a.output,a.epochs,device=a.device))
