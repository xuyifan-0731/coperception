#!/usr/bin/env python
import os, glob, argparse, wandb
import numpy as np
import torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# ======== ARGPARSE ========
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True)
p.add_argument('--data_root_val', default=None)
p.add_argument('--T', type=int, default=5)
p.add_argument('--n', type=int, default=1)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--eval_interval', type=int, default=100)
p.add_argument('--use_aug', action='store_true')
p.add_argument('--static_sampling_ratio', type=float, default=1.0,
               help='Fraction of static pixels to sample (0-1)')
p.add_argument('--curriculum_epochs', type=int, default=5,
               help='Epochs before enabling dynamic losses')
p.add_argument('--use_dyn_weight', action='store_true')
p.add_argument('--use_change_head', action='store_true')
p.add_argument('--use_dice', action='store_true')
p.add_argument('--use_focal', action='store_true')
p.add_argument('--use_contrast', action='store_true')
p.add_argument('--lambda_d', type=float, default=4.0)
p.add_argument('--alpha_dice', type=float, default=1.0)
p.add_argument('--alpha_chg', type=float, default=0.5)
p.add_argument('--alpha_focal', type=float, default=1.0)
p.add_argument('--alpha_contrast', type=float, default=0.1)
p.add_argument('--project', default='bev-full-opt')
p.add_argument('--name', default='run1')
args = p.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== UTILS ========
def sparse2bev(idx, shape):
    bev = np.zeros(shape, np.uint8)
    y,x = idx[:,1], idx[:,2]
    m = (y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])
    bev[y[m],x[m]] = 1
    return bev

def iou(pred, true):
    p = pred.cpu().numpy().astype(bool)
    t = true.cpu().numpy().astype(bool)
    inter = np.logical_and(p,t).sum()
    union= np.logical_or(p,t).sum()
    return inter/union if union>0 else 1

def iou_batch(pred, true):
    return np.mean([iou(pred[b,0], true[b,0]) for b in range(pred.shape[0])])

def prfa(pred, true):
    p=precision_score(true.flatten(),pred.flatten(),zero_division=1)
    r=recall_score(true.flatten(),pred.flatten(),zero_division=1)
    f=f1_score(true.flatten(),pred.flatten(),zero_division=1)
    a=(pred.flatten()==true.flatten()).mean()
    return p,r,f,a

def dice_loss(pred, target, eps=1e-6):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p*t).sum(1)
    union = p.sum(1)+t.sum(1)
    return (1 - (2*inter+eps)/(union+eps)).mean()

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p_t = pred*target + (1-pred)*(1-target)
    alpha_t = alpha*target + (1-alpha)*(1-target)
    loss = -alpha_t * (1-p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-8))
    return loss.mean()

# ======== DATASET ========
class SeqDS(Dataset):
    def __init__(self, root,T,n,aug=False):
        self.root, self.T, self.n, self.aug = root,T,n,aug
        self.by_scene={}
        for f in glob.glob(root+'/*_*'):
            sc,fr=map(int,os.path.basename(f).split('_'))
            self.by_scene.setdefault(sc,[]).append(fr)
        for sc in self.by_scene: self.by_scene[sc].sort()
        for sc,frs in self.by_scene.items():
            if len(frs)>=T+n:
                d=np.load(f'{root}/{sc}_{frs[0]}/0.npy',allow_pickle=True).item()
                idx=d['voxel_indices_0']; self.H, self.W = idx[:,1].max()+1, idx[:,2].max()+1
                break
    def __len__(self):
        return sum(max(0, len(frs)-self.T-self.n+1) for frs in self.by_scene.values())
    def __getitem__(self,i):
        cum=0
        for sc,frs in self.by_scene.items():
            L=max(0,len(frs)-self.T-self.n+1)
            if i<cum+L:
                st=i-cum; hist=frs[st:st+self.T]; tgt=frs[st+self.T+self.n-1]
                X=[]
                for fr in hist:
                    d=np.load(f'{self.root}/{sc}_{fr}/0.npy',allow_pickle=True).item()
                    X.append(sparse2bev(d['voxel_indices_0'],(self.H,self.W)))
                d=np.load(f'{self.root}/{sc}_{tgt}/0.npy',allow_pickle=True).item()
                Y=sparse2bev(d['voxel_indices_0'],(self.H,self.W))
                if self.aug:
                    flip=np.random.rand()<0.5; rot=np.random.choice([0,2])
                    for j in range(len(X)):
                        if flip: X[j]=np.fliplr(X[j])
                        X[j]=np.rot90(X[j],rot).copy()
                    if flip: Y=np.fliplr(Y)
                    Y=np.rot90(Y,rot).copy()
                X=torch.from_numpy(np.stack(X)).unsqueeze(1).float()
                Y=torch.from_numpy(Y).unsqueeze(0).float()
                return X,Y
            cum+=L
        raise IndexError

# ======== MODEL ========
class CLSTMCell(nn.Module):
    def __init__(s,c_in,c_hid):
        super().__init__()
        s.conv=nn.Conv2d(c_in+c_hid,4*c_hid,3,1,1)
        s.c_hid=c_hid
    def forward(s,x,h,c):
        g=s.conv(torch.cat([x,h],1)); i,f,o,gg=g.chunk(4,1)
        i,f,o=torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o)
        gg=torch.tanh(gg); c2=f*c+i*gg; h2=o*torch.tanh(c2)
        return h2,c2

class CLSTM(nn.Module):
    def __init__(s): super().__init__(); s.cell=CLSTMCell(1,32)
    def forward(s,x):
        B,T,_,H,W=x.shape
        h=x.new_zeros(B,32,H,W); c=x.new_zeros(B,32,H,W)
        for t in range(T): h,c=s.cell(x[:,t],h,c)
        return h

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.back=CLSTM()
        s.head_occ=nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.ReLU(),
                                  nn.Conv2d(32,1,1),nn.Sigmoid())
        if args.use_change_head:
            s.head_chg=nn.Sequential(nn.Conv2d(32,16,3,1,1),nn.ReLU(),
                                      nn.Conv2d(16,1,1),nn.Sigmoid())
    def forward(s,x):
        f=s.back(x)
        occ=s.head_occ(f)
        if args.use_change_head:
            chg=s.head_chg(f)
            return occ, chg
        return occ, None

class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, k=3):
        super().__init__()
        p = k//2
        self.conv = nn.Conv2d(c_in + c_hid, 4*c_hid, k, padding=p)
        self.c_hid = c_hid

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class UNetConvLSTM(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, T=5, use_change_head=False):
        super().__init__()
        self.T = T
        self.use_change_head = use_change_head
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, 1, 1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        # ConvLSTM cells
        self.clstm1 = ConvLSTMCell(base_ch*2, base_ch*2)
        self.clstm2 = ConvLSTMCell(base_ch*3, base_ch)
        # Decoder
        self.dec0 = nn.Sequential(nn.Conv2d(base_ch+in_ch, base_ch, 3, 1, 1), nn.ReLU())
        self.head_occ = nn.Conv2d(base_ch, 1, 1)
        if use_change_head:
            self.head_chg = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        B, T, _, H, W = x.shape
        f1_list, f2_list = [], []
        for t in range(T):
            xt = x[:, t]
            e1 = self.enc1(xt)
            e2 = self.enc2(self.pool(e1))
            f1_list.append(e1)
            f2_list.append(e2)
        # Bottleneck
        h2 = torch.zeros(B, f2_list[0].shape[1], f2_list[0].shape[2], f2_list[0].shape[3], device=x.device)
        c2 = h2.clone()
        for t in range(T):
            h2, c2 = self.clstm1(f2_list[t], h2, c2)
        u2 = F.interpolate(h2, size=(H, W), mode='bilinear', align_corners=False)
        # Level1
        h1 = torch.zeros(B, f1_list[0].shape[1], H, W, device=x.device)
        c1 = h1.clone()
        for t in range(T):
            e1 = f1_list[t]
            if u2.shape[-2:] != e1.shape[-2:]:
                u2 = F.interpolate(u2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            inp = torch.cat([e1, u2], dim=1)
            h1, c1 = self.clstm2(inp, h1, c1)
        # Decoder
        u1 = F.interpolate(h1, scale_factor=2, mode='bilinear', align_corners=False)
        last = x[:, -1]
        if u1.shape[-2:] != last.shape[-2:]:
            u1 = F.interpolate(u1, size=last.shape[-2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([u1, last], dim=1))
        occ = torch.sigmoid(self.head_occ(d0))
        chg = None
        if self.use_change_head:
            chg = torch.sigmoid(self.head_chg(d0))
        return occ, chg

# ======== TRAIN/EVAL ========
def evaluate(model,loader,epoch,step):
    model.eval()
    iop,iob,di,dp,ps,rs,fs,acs = [],[],[],[],[],[],[],[]
    with torch.no_grad():
        for X,Y in loader:
            X,Y=X.to(DEVICE),Y.to(DEVICE)
            occ, chg=model(X)
            pv=(occ>0.5).float(); base=(X[:,-1]>0.5).float()
            iop.append(iou_batch(pv,Y)); iob.append(iou_batch(base,Y))
            di.append(iop[-1]-iob[-1])
            # dyn IoU
            M=(base!=Y).float()
            p_dyn=(pv*M).cpu().numpy().astype(bool)[:,0]
            t_dyn=(Y*M).cpu().numpy().astype(bool)[:,0]
            dp.append(np.mean([ (p_dyn[b]&t_dyn[b]).sum()/
                     (p_dyn[b]|t_dyn[b]).sum() if (p_dyn[b]|t_dyn[b]).sum()>0 else 1
                     for b in range(p_dyn.shape[0])]))
            # prfa
            pr,rc,f1,ac=prfa(pv.cpu().numpy(),Y.cpu().numpy())
            ps.append(pr); rs.append(rc); fs.append(f1); acs.append(ac)
    wandb.log({
        'val/IoU':np.mean(iop),'val/Baseline_IoU':np.mean(iob),
        'val/Delta_IoU':np.mean(di),'val/Dynamic_IoU':np.mean(dp),
        'val/P':np.mean(ps),'val/R':np.mean(rs),
        'val/F1':np.mean(fs),'val/Acc':np.mean(acs),
        'epoch':epoch,'step':step})
    model.train()

def train():
    wandb.init(project=args.project,entity=None,config=vars(args),name=args.name)
    root_val=args.data_root_val or args.data_root
    ds_tr=SeqDS(args.data_root,args.T,args.n,args.use_aug)
    ds_va=SeqDS(root_val,args.T,args.n,False)
    trl=DataLoader(ds_tr,args.batch,True)
    val=DataLoader(ds_va,args.batch,False)
    net=UNetConvLSTM().to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)
    step=0
    for ep in range(1,args.epochs+1):
        net.train()
        for X,Y in trl:
            X,Y=X.to(DEVICE),Y.to(DEVICE)
            occ,chg=net(X)
            # masks
            prev=X[:,-1]; dyn=(prev!=Y).float()
            # losses
            loss_bce=F.binary_cross_entropy(occ,Y)
            if args.static_sampling_ratio<1:
                stat=(dyn==0)
                idx=torch.nonzero(stat.view(-1),as_tuple=False)
                keep=int(len(idx)*args.static_sampling_ratio)
                m=torch.zeros_like(dyn.view(-1))
                m[idx[torch.randperm(len(idx))[:keep]]] = 1
                m=m.view_as(dyn); weight=1+args.lambda_d*dyn*m
                loss_bce=F.binary_cross_entropy(occ,Y,weight=weight)
            elif args.use_dyn_weight:
                weight=1+args.lambda_d*dyn
                loss_bce=F.binary_cross_entropy(occ,Y,weight=weight)
            loss=loss_bce
            if args.use_dice and ep>args.curriculum_epochs:
                loss+=args.alpha_dice*dice_loss(occ,Y)
            if args.use_change_head and chg is not None and ep>args.curriculum_epochs:
                loss+=args.alpha_chg*F.binary_cross_entropy(chg,dyn)
            if args.use_focal and args.use_change_head and ep>args.curriculum_epochs:
                print("chg,dyn",chg,dyn)
                # TODO: chg is none here
                loss+=args.alpha_focal*focal_loss(chg,dyn)
            if args.use_contrast:
                loss+=args.alpha_contrast*F.mse_loss(occ,prev)
            opt.zero_grad(); loss.backward(); opt.step()
            wandb.log({'train/loss':loss.item(),'epoch':ep,'step':step})
            if step and step%args.eval_interval==0: evaluate(net,val,ep,step)
            step+=1
    wandb.finish()

if __name__=='__main__':
    train()
