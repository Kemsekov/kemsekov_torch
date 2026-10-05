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
        # TorchScript indexes strings by UTF-8 byte, which breaks on multibyte
        # symbols, so keep an explicit list of characters for decoding
        self.idx2sym_list: List[str] = [s for s in self.idx2sym]
        
        
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
        chars = [self.idx2sym_list[i] for i in indices]
        return ''.join(chars)

class HFTokenizer(nn.Module):
    """
    Drop-in replacement for SimpleTokenizer backed by a HuggingFace `tokenizers` tokenizer
    trained on the provided text lines.

    `hf_tokenizer_type` selects the algorithm ("bpe", "wordpiece" or "unigram"),
    the other hf_* arguments configure training. The fitted tokenizer is kept as its
    JSON config in `hf_config`, so the module stays torch.jit exportable and can be
    restored after `torch.jit.load` without this class:

        tok = torch.jit.load("tokenizer.pt")
        from tokenizers import Tokenizer
        hf_tokenizer = Tokenizer.from_str(tok.hf_config)
    """
    def __init__(self, texts: List[str], lowercase=False, unknown_symbols_placeholder=' ',
                 hf_tokenizer_type="bpe", hf_vocab_size=32000, hf_min_frequency=2,
                 hf_special_tokens: List[str] = None):
        super().__init__()
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        if hf_tokenizer_type == "bpe":
            hf_tokenizer = Tokenizer(models.BPE())
            hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            hf_tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=hf_vocab_size,
                min_frequency=hf_min_frequency,
                special_tokens=hf_special_tokens or [],
                show_progress=False,
            )
        elif hf_tokenizer_type == "wordpiece":
            hf_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            hf_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            hf_tokenizer.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=hf_vocab_size,
                min_frequency=hf_min_frequency,
                special_tokens=hf_special_tokens or ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=False,
            )
        elif hf_tokenizer_type == "unigram":
            special_tokens = hf_special_tokens or ["[UNK]"]
            if "[UNK]" not in special_tokens:
                special_tokens = ["[UNK]"] + special_tokens
            hf_tokenizer = Tokenizer(models.Unigram())
            hf_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
            hf_tokenizer.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(
                vocab_size=hf_vocab_size,
                unk_token="[UNK]",
                special_tokens=special_tokens,
                show_progress=False,
            )
        else:
            raise ValueError("Unknown hf_tokenizer_type '%s', expected bpe/wordpiece/unigram" % hf_tokenizer_type)

        hf_tokenizer.train_from_iterator(
            (t.lower() for t in texts) if lowercase else texts,
            trainer=trainer,
        )

        self.hf_tokenizer_type = hf_tokenizer_type
        self.hf_vocab_size = hf_vocab_size
        self.hf_config = hf_tokenizer.to_str()
        self._hf_tokenizer = hf_tokenizer  # python-only, not serialized by torch.jit

        self.vocab_size = hf_tokenizer.get_vocab_size(with_added_tokens=True)
        self.idx2sym_list: List[str] = [
            hf_tokenizer.id_to_token(i) or unknown_symbols_placeholder
            for i in range(self.vocab_size)
        ]
        self.idx2sym: torch.StringType = "".join(self.idx2sym_list)
        self.sym2idx: Dict[str, int] = hf_tokenizer.get_vocab()
        self.unknown_symbols_placeholder = unknown_symbols_placeholder
        self.lowercase = lowercase

        placeholder_ids = hf_tokenizer.encode(unknown_symbols_placeholder).ids
        self.space_idx = placeholder_ids[0] if len(placeholder_ids) > 0 else 0

    def load(path_to_file: str):
        from tokenizers import Tokenizer
        tok : nn.Module = torch.jit.load(path_to_file)
        hf_tokenizer = Tokenizer.from_str(tok.hf_config)
        res = HFTokenizer(["1"])
        res.load_state_dict(tok.state_dict())
        
        # do this for all attributes of  res.attr=tok.attr
        for k in dir(res):
            if not k.startswith('_') and not callable(getattr(res, k)):
                setattr(res, k, getattr(tok, k))
        
        res._hf_tokenizer=hf_tokenizer
        return res
    
    def forward(self, x):
        return x

    def encode(self, text: str) -> torch.Tensor:
        """Convert a string to a tensor of token indices (torch.long)."""
        if self.lowercase:
            text = text.lower()
        return torch.tensor(self._hf_tokenizer.encode(text).ids, dtype=torch.long)

    def decode(self, indices: Tensor) -> str:
        """Convert a tensor of token indices back to a string."""
        return self._hf_tokenizer.decode(indices.tolist())


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
        if tokenizer.vocab_size<256:
            self.store_dtype=torch.uint8
        elif tokenizer.vocab_size<65536:
            self.store_dtype=torch.uint16
        else:
            self.store_dtype=torch.uint32
    
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



# module to convert text tokens to vector
class Embedding(nn.Module):
    """
    Module for token to embedding vector learning
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.bias = nn.Parameter(torch.Tensor(embedding_size))

        self.reset_parameters()

    #normal init
    def reset_parameters(self):
        # Initialize weights with a normal distribution
        std = 1.0 / (self.vocab_size**0.5)
        
        nn.init.normal_(self.weight, mean=0.0, std=std)
        # Initialize bias to zeros
        nn.init.zeros_(self.bias)
        
    def forward(self, input):
        # Input is expected to be a tensor of indices
        return torch.nn.functional.embedding(input, self.weight)

    def encode(self,ind): return self(ind)
    
    def decode(self,act):
        return act@self.weight.T