import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);self.encoder=nn.Sequential(nn.Linear(3,w),nn.SiLU(),nn.Linear(w,w));self.decoder=nn.Sequential(nn.Linear(w+2,w),nn.SiLU(),nn.Linear(w,1))
    def reconstruct(self,x,available,query):
        token=self.encoder(x);summary=(token*available[:,:,None]).sum(1)/available.sum(1,keepdim=True).clamp_min(1)
        return self.decoder(torch.cat([query,summary[:,None].expand(-1,query.shape[1],-1)],-1)).squeeze(-1)
    def forward(self,b):
        x=b['x'].float();available=b['available'].float();query=b['query'].float();mean=self.reconstruct(x,available,query);prior=self.reconstruct(x,torch.zeros_like(available),query)
        removals=[]
        for sensor in range(x.shape[1]):
            mask=available.clone();mask[:,sensor]=0;removals.append(abs(self.reconstruct(x,mask,query)-mean))
        removal=torch.stack(removals,1);scale=self.c.get('temperature_scale',1.);gain=abs(mean-prior)/scale;fragility=removal.square().mean(1).sqrt()/scale
        if self.c['variant']=='no_gain':gain=torch.zeros_like(gain)
        if self.c['variant']=='no_removal':fragility=torch.zeros_like(fragility)
        trust=(fragility+self.c.get('tau',.01))/(gain+fragility+self.c.get('tau',.01));radius=scale*trust
        if self.c['variant']=='baseline':radius=torch.ones_like(radius)*scale
        return {'prediction':mean,'lower':mean-radius,'upper':mean+radius,'trust_risk':trust,'gain':gain,'fragility':fragility,'influence':removal.square()/removal.square().sum(1,keepdim=True).clamp_min(1e-8)}
    def objective(self,b,p):return F.l1_loss(p['prediction'],b['y'].float())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);sensor=rng.uniform(0,1,(n,8,2)).astype('float32');query=rng.uniform(0,1,(n,64,2)).astype('float32');temperature=lambda z:20+5*z[...,0]+3*z[...,1]**2
    return {'x':np.concatenate([sensor,temperature(sensor)[...,None]],-1).astype('float32'),'available':np.ones((n,8),np.float32),'query':query,'y':temperature(query).astype('float32'),'group':np.arange(n)}


def measure(p,d,c):
    error=abs(p['prediction']-d['y']);order=np.argsort(p['trust_risk'].flatten());e=error.flatten()[order];risk=np.cumsum(e)/np.arange(1,len(e)+1)
    return {'aurc':float(risk.mean()),'risk_at_80pct':float(e[:max(1,int(.8*len(e)))].mean()),'mean_evidence_gain':float(p['gain'].mean()),'mean_removal_fragility':float(p['fragility'].mean())}
