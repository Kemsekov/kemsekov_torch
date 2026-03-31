from typing import Literal
import torch
import torch.nn as nn

class LRU(nn.Module):
    """
    Linear recurrent unit
    
    """
    def __init__(
        self,
        dim,
        c=4,
        expected_sequence_length = 512,
        implementation : Literal['loop','log-exp','a-scan']='log-exp'
    ):
        """
        dim: timeseries dimensions
        c: increasing this value will shorten LRU memory (it will forget faster)
        expected_sequence_length: seq length that we expect to see in training
        implementation: which implementation to use for recurrence
        """
        super().__init__()
        self.linear_gate = nn.Linear(dim,dim*2)
        
        # this decay works as forgetting gate, and we make sure to
        # start with small decay so we keep the memory
        
        target_alpha = 1.0 - (1.0 / expected_sequence_length)
        initial_decay = torch.logit(torch.tensor(target_alpha**(1/c)))
        self.decay = nn.Parameter(torch.ones((dim,))*initial_decay)
        
        # self.decay = torch.nn.Parameter(torch.empty(dim).uniform_(-4, 0))
        self.start_state = torch.nn.Parameter(torch.randn((dim,)))
        
        self.c=c
        self.implementation=implementation
        
    def forward(self,x):
        # x of shape [batch,seqlen,dim]
        rt,it = self.linear_gate(x).sigmoid().chunk(2,-1)
        
        at = self.decay.sigmoid()**(rt*self.c)
        # this stuff is needed for variance-preserving
        sqrt_one_minus_at2 = torch.sqrt(1-at**2+1e-6)
        itx = sqrt_one_minus_at2*it*x
        
        # the resulting tensor
        if self.implementation=='loop':
            return self.loop(at, itx)
        if self.implementation=='log-exp':
            return self.log_exp_trick(at, itx)
        return self.associative_scan_db(at, itx)

    def log_exp_trick(self, at: torch.Tensor, itx: torch.Tensor):
        # 1. Compute the log of the decays for numerical stability
        # at shape: [batch, seqlen, dim]
        log_at = torch.log(at.clip(1e-7)) 
        
        # 2. Compute the cumulative sum of logs (this represents the product a_1 * a_2 * ...)
        a_star = torch.cumsum(log_at, dim=1)
        
        # 3. The "Decay-Corrected" inputs
        exp_a_star = torch.exp(a_star.clip(max=70))
        
        # Calculate the contribution of the start_state
        h_0_part = exp_a_star * self.start_state.unsqueeze(0)
        
        # Calculate the contribution of the inputs (itx)
        # We use a small epsilon to avoid division by zero
        inner_term = itx / (exp_a_star + 1e-9)
        x_part = exp_a_star * torch.cumsum(inner_term, dim=1)
        
        return h_0_part + x_part
    
    def loop(self, at, itx):
        seqlen = itx.shape[1]
        hid = torch.zeros_like(itx)
        for i in range(seqlen):
            prev_h = hid[:,i-1] if i>0 else self.start_state
            h_curr = at[:, i] * prev_h + itx[:, i]
            hid[:, i] = h_curr
            
        return hid
    # double-buffer associative scan
    def associative_scan_db(self, at, itx):
        batch, seq_len, dim = at.shape
        
        # 1. Initialize our first state
        # We start by integrating the start_state into the first token
        res_x = itx.clone()
        res_x[:, 0] = res_x[:, 0] + at[:, 0] * self.start_state
        res_a = at
        
        i = 1
        while i < seq_len:
            # Create NEW tensors for this jump (no in-place modification)
            # We copy the 'head' (0:i) as-is and compute the 'tail' (i:)
            next_a = res_a.clone()
            next_x = res_x.clone()

            
            # Slicing is efficient and doesn't trigger the in-place error
            a_tail = res_a[:, i:]
            x_left = res_x[:, :-i]
            a_left = res_a[:, :-i]
            
            next_x[:, i:] = res_x[:, i:] + a_tail * x_left
            next_a[:, i:] = a_tail * a_left
            
            # Move to the next power-of-two state
            # This 'swap' is safe because next_x/a are fresh tensors
            res_x, res_a = next_x, next_a
            i *= 2
            
        return res_x