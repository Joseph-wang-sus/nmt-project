import re
import json
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple, Union

import torch
import hanlp
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIALS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# ==========================================
# 1. Data Cleaning
# ==========================================

def clean_text(text: str) -> str:
    """Normalizes whitespace and removes non-printable control characters."""
    if not text:
        return ""
    # Remove control characters (except tabs/newlines if you want to keep them, here we strip all)
    text = "".join(ch for ch in text if ch.isprintable())
    # Normalize whitespace (tabs, multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_example(
    src_text: str, 
    trg_text: str, 
    min_len: int = 1, 
    max_len: int = 128, 
    truncate: bool = False
) -> Optional[Tuple[str, str]]:
    """
    Filters or truncates a pair of sentences.
    Returns None if the pair should be dropped.
    """
    src = clean_text(src_text)
    trg = clean_text(trg_text)

    if not src or not trg:
        return None

    # Simple approximate length check (by character for simplicity at this stage, 
    # or strict token count later. Here we use char count/rough split for speed)
    src_len = len(src.split())  # Approximation
    trg_len = len(trg.split())

    if src_len < min_len or trg_len < min_len:
        return None

    if truncate:
        # Note: True truncation usually happens AFTER tokenization. 
        # Here we just pass through if we allow long sentences, 
        # assuming the collator handles truncation, or we drop them.
        pass 
    else:
        if src_len > max_len or trg_len > max_len:
            return None

    return src, trg

# ==========================================
# 2. Tokenizers
# ==========================================

class BaseTokenizer:
    def encode(self, text: str) -> List[str]:
        raise NotImplementedError
    
    def decode(self, tokens: List[str]) -> str:
        raise NotImplementedError

class ChineseTokenizerHanLP(BaseTokenizer):
    """
    Wrapper for HanLP Chinese Word Segmentation.
    """
    def __init__(self, task: str = 'tok/coarse'):
        logger.info(f"Loading HanLP model: {task}...")
        # Ensure you have downloaded the model or allow hanlp to download it
        self.pipeline = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH) 
    
    def encode(self, text: str) -> List[str]:
        # HanLP returns a list of strings
        try:
            return self.pipeline(text)
        except Exception as e:
            logger.warning(f"HanLP error on text '{text}': {e}")
            return [UNK_TOKEN]

    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)

class BPETokenizerEn(BaseTokenizer):
    """
    Wrapper for Hugging Face Tokenizers (BPE) implementation.
    """
    def __init__(self, vocab_size: int = 30000, model_path: Optional[str] = None):
        if model_path:
            self.tokenizer = Tokenizer.from_file(model_path)
        else:
            # Initialize a fresh BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            decoder_cls = getattr(decoders, "BPE", None)
            if decoder_cls is not None:
                self.tokenizer.decoder = decoder_cls()
            else:
                logger.warning("tokenizers.decoders.BPE missing in this version; using default decoder.")
            self.vocab_size = vocab_size

    def train(self, files: List[str]):
        """Trains BPE on a list of text files."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=SPECIALS,
            show_progress=True
        )
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[str]:
        # Returns list of token strings (subwords)
        return self.tokenizer.encode(text).tokens

    def decode(self, tokens: List[str]) -> str:
        # Simple join, assuming BPE decoder handles ## or similar artifacts if configured
        # But HF tokenizer decode expects IDs mostly. We use manual join for simplicity here
        # or convert to IDs and use internal decoder.
        # For NMT course simplicity: just join with space and let BPE decoder handle specifics if needed.
        return self.tokenizer.decode(self.tokenizer.token_to_id(t) for t in tokens if t in self.tokenizer.get_vocab())
    
    def save(self, path: str):
        self.tokenizer.save(path)

# ==========================================
# 3. Vocabulary Construction
# ==========================================

class Vocab:
    def __init__(self, tokens_iterator=None, min_freq: int = 1, specials: List[str] = SPECIALS):
        self.stoi = {}
        self.itos = []
        self.specials = specials
        
        # Add specials first
        for s in specials:
            self._add_token(s)
            
        if tokens_iterator:
            counter = Counter()
            for tokens in tokens_iterator:
                counter.update(tokens)
            
            # Sort by frequency (high to low)
            sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            
            for token, freq in sorted_by_freq:
                if freq >= min_freq:
                    self._add_token(token)

        self.unk_index = self.stoi.get(UNK_TOKEN, 0)
        self.pad_index = self.stoi.get(PAD_TOKEN, 1)

    def _add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_index)

    def encode(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = [self[t] for t in tokens]
        if add_bos:
            ids = [self.stoi[BOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.stoi[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        tokens = []
        for i in ids:
            token = self.itos[i] if i < len(self.itos) else UNK_TOKEN
            if remove_special and token in self.specials:
                continue
            tokens.append(token)
        return tokens

# ==========================================
# 4. Word Embedding Initialization
# ==========================================

def load_pretrained_embeddings(
    vocab: Vocab, 
    path_to_vec: str, 
    emb_dim: int
) -> torch.Tensor:
    """
    Loads pretrained vectors (e.g. GloVe .txt) and initializes an embedding matrix.
    Format expected: "token val1 val2 ..."
    """
    logger.info(f"Loading embeddings from {path_to_vec}...")
    
    # Random initialization (fallback)
    embedding_matrix = torch.normal(0, 0.1, size=(len(vocab), emb_dim))
    
    hits = 0
    with open(path_to_vec, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = parts[1:]
            
            if word in vocab.stoi:
                if len(vector) == emb_dim:
                    embedding_matrix[vocab.stoi[word]] = torch.tensor([float(x) for x in vector])
                    hits += 1
                else:
                    logger.warning(f"Dimension mismatch for word '{word}': expected {emb_dim}, got {len(vector)}")

    logger.info(f"Loaded {hits}/{len(vocab)} vectors from pretrained file.")
    return embedding_matrix