import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict

class SimpleTokenizer(nn.Module):
    def __init__(self, texts: List[str], lowercase=False,unknown_symbols_placeholder = ' '):
        super().__init__()
        # Collect all unique symbols
        unique_symbols = set()
        for text in texts:
            t = text
            if lowercase:
                t = t.lower()
            unique_symbols.update(t)
        unique_symbols.add(unknown_symbols_placeholder)  # Ensure space is always included as fallback
        self.idx2sym : torch.StringType = "".join(sorted(unique_symbols))
        
        
        
        self.sym2idx: Dict[str, int] = {s: i for i, s in enumerate(self.idx2sym)}
        self.unknown_symbols_placeholder=unknown_symbols_placeholder
        self.space_idx = self.sym2idx[unknown_symbols_placeholder]  # Used for unknown characters
        self.lowercase = lowercase
        self.vocab_size = len(self.idx2sym)
        
        txt_split = [b for v in texts if len(v)>30 for b in v.split('\n')]
        lengths = [len(v) for v in txt_split]
        lengths=np.array(lengths)

        print("Text length analysis")
        print("text lines\t",len(txt_split))
        print("line chars mean\t",lengths.mean().round(3))
        print("line chars std\t",lengths.std().round(3))
        print("0.05 quantile\t",np.quantile(lengths,0.05))
        print("0.95 quantile\t",np.quantile(lengths,0.95))
        print("0.995 quantile\t",np.quantile(lengths,0.995))

    def forward(self, x):
        return x

    @torch.jit.export
    def encode(self, text: str) -> torch.Tensor:
        """
        Convert a string to a tensor of indices (torch.long), using space index for unknown symbols.
        """
        if self.lowercase:
            text = text.lower()
        idxs = [self.sym2idx.get(ch, self.space_idx) for ch in text]
        return torch.tensor(idxs)

    @torch.jit.export
    def decode(self, indices: Tensor) -> str:
        """
        Convert a tensor of indices back to a string.
        """
        chars = [self.idx2sym[i] for i in indices]
        return ''.join(chars)

class TokenDataset(torch.utils.data.Dataset):
    """
    Dataset that returns tokenized text with output tokens length as multiple of `batch_size`.
    Use this dataset alongside kemsekov_torch.utils.BinBySizeDataset and with model `torch.compile(dynamic=True)`
    """
    
    def __init__(
        self, 
        tokenizer : SimpleTokenizer, 
        text_lines,pad_token = ' ',
        batch_size = 64,
        max_length = 1024,
        fixed_length=None
    ):
        super().__init__()
        text_lines=[t.strip() for t in text_lines]
        self.text = [t for t in text_lines if len(t)>0]
        self.pad_token = list(tokenizer.encode(pad_token))
        self.batch_size=batch_size
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.cache = {}
        self.fixed_length=fixed_length
        
        # to save memory, store cache in lowest precision
        if len(tokenizer.idx2sym)<256:
            self.store_dtype=torch.uint8
        else:
            self.store_dtype=torch.uint16
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index].long()
        text = self.text[index]
        ids_orig = self.tokenizer.encode(text)
        true_text = ids_orig[:self.max_length-1]
        if self.fixed_length is None:
            output_tokens = int(math.ceil(len(true_text)/self.batch_size)*self.batch_size)
        else:
            output_tokens=self.fixed_length
        ids=torch.tensor(list(true_text)+self.pad_token*(output_tokens))[:output_tokens]
        self.cache[index]=ids.to(self.store_dtype)
        return ids