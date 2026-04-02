import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DensityRegressor(nn.Module):
    def __init__(self, in_dim,scale=(-3,3),hid=128,bins=32,dropout=0) -> None:
        super().__init__()
        self.scale=(min(scale)-1e-8,max(scale)+1e-8)
        self.bins=bins
        self.m = nn.Sequential(
            nn.Linear(in_dim,hid),
            nn.RMSNorm(hid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid,hid),
            nn.RMSNorm(hid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid,bins),
        )
    
    def forward(self,x) -> torch.Tensor:
        return self.m(x).log_softmax(-1)
    
    def log_prob(self,x,y,cumsum=False):
        logp = self.forward(x)
        # move y to our scale
        y=((y-self.scale[0])/(self.scale[1]-self.scale[0]))
        ind = (y*self.bins).long().clip(0,self.bins-1)
        
        if cumsum:
            logp = logp.logcumsumexp(-1)
        
        return logp[torch.arange(len(ind),device=x.device),ind]
    
    def predict_proba(self,x):
        logp = self.forward(x)
        values = torch.linspace(
            self.scale[0],
            self.scale[1],
            self.bins,
            device=x.device
        )
        return values,logp
    
    def predict_mean(self,x):
        values,logp = self.predict_proba(x)
        mean_prediction = (values*logp.exp()).sum(-1)
        return mean_prediction
        

class ContinuousDensityRegressor(nn.Module):
    def __init__(self, in_dim,scale=(-3,3),hid=128,bins=32,dropout=0) -> None:
        super().__init__()
        self.scale=(min(scale)-1e-8,max(scale)+1e-8)
        self.bins=bins
        self.m = nn.Sequential(
            nn.Linear(in_dim,hid),
            nn.RMSNorm(hid),
            # nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hid,hid),
        )
        self.y_density_hid = nn.Sequential(
            nn.Linear(1,hid),
            # nn.RMSNorm(hid),
            # nn.SiLU(),
            # nn.Linear(hid,hid),
        )
        self.density = nn.Sequential(
            nn.Linear(hid,hid),
            nn.RMSNorm(hid),
            # nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hid,hid),
            nn.RMSNorm(hid),
            # nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hid,1),
        )
    
    def forward(self,x,bins=None) -> torch.Tensor:
        if bins is None: bins=self.bins
        
        # [batch,hid]
        hid = self.m(x)
        
        y = torch.linspace(self.scale[0],self.scale[1],bins,device=x.device)[:,None]
        
        # [bins,hid]
        y_hid = self.y_density_hid(y)
        
        #[batch,bins,hid]
        combine = hid[:,None]+y_hid[None,:]
        
        #[batch,bins]
        density : torch.Tensor = self.density(combine)[:,:,0]
        
        # compute normalization constant
        # norm = density.logsumexp(-1)#-math.log(dt)
        # return density-norm[:,None]
        return density.log_softmax(-1)
    
    def log_prob(self,x,y,cumsum=False,bins=None):
        if bins is None: bins=self.bins
        logp = self.forward(x,bins)
        # move y to our scale
        y=((y-self.scale[0])/(self.scale[1]-self.scale[0]))
        ind = (y*bins).long().clip(0,bins-1)
        
        if cumsum:
            logp = logp.logcumsumexp(-1)
        
        return logp[torch.arange(len(ind),device=x.device),ind]
    
    def predict_proba(self,x,bins = None):
        if bins is None: bins=self.bins
        logp = self.forward(x,bins)
        values = torch.linspace(
            self.scale[0],
            self.scale[1],
            bins,
            device=x.device
        )
        return values,logp
    
    def predict_mean(self,x,bins = None):
        values,logp = self.predict_proba(x,bins)
        mean_prediction = (values*logp.exp()).sum(-1)
        return mean_prediction
        