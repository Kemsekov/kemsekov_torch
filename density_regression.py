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
    
    def predict(self,x):
        logp = self.forward(x)
        values = torch.linspace(
            self.scale[0],
            self.scale[1],
            self.bins,
            device=x.device
        )
        return values,logp
    