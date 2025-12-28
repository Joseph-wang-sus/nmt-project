import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    vocab_size_src: int = 10000
    vocab_size_tgt: int = 10000
    max_seq_len: int = 128
    
    # === [Key Fix] Must add pad_idx definition ===
    pad_idx: int = 0  
    # ======================================
    
    # Ablation settings
    pos_embedding_type: str = "absolute" # "absolute" or "relative"
    norm_type: str = "layernorm"         # "layernorm" or "rmsnorm"

# ==========================================
# 1. Normalization Modules
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

def get_norm(type_str, d_model):
    if type_str == "rmsnorm":
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)

# ==========================================
# 2. Position Embeddings
# ==========================================

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class RelativePositionBias(nn.Module):
    """Simple T5-style relative position bias.
    Ensure `max_distance` covers maximum sequence lengths to avoid OOB indices.
    """
    def __init__(self, num_heads, max_distance: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = int(max_distance)
        self.embeddings = nn.Embedding(self.max_distance * 2 + 1, num_heads)

    def forward(self, seq_len_q, seq_len_k):
        # Create a matrix of relative positions
        range_vec_q = torch.arange(seq_len_q, dtype=torch.long, device=self.embeddings.weight.device)
        range_vec_k = torch.arange(seq_len_k, dtype=torch.long, device=self.embeddings.weight.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None] 
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance)
        final_mat = (distance_mat_clipped + self.max_distance).to(torch.long)  # [0, 2*max_dist]
        
        # [seq_len_q, seq_len_k, num_heads] -> [1, num_heads, seq_len_q, seq_len_k]
        bias = self.embeddings(final_mat)
        return bias.permute(2, 0, 1).unsqueeze(0)

# ==========================================
# 3. Core Transformer Components
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_k = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        
        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.out_linear = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None, rel_pos_bias=None):
        bs = q.size(0)
        
        # Linear projections & split heads
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if rel_pos_bias is not None:
            # bias shape: [1, n_heads, seq_len_q, seq_len_k]
            scores += rel_pos_bias
            
        if mask is not None:
            # mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        
        # Concat heads
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        return self.out_linear(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = get_norm(config.norm_type, config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = get_norm(config.norm_type, config.d_model)
        self.ff = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask, rel_pos_bias=None):
        # Pre-Norm Architecture
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, x2, x2, mask, rel_pos_bias))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = get_norm(config.norm_type, config.d_model)
        self.self_attn = MultiHeadAttention(config)
        self.norm2 = get_norm(config.norm_type, config.d_model)
        self.cross_attn = MultiHeadAttention(config)
        self.norm3 = get_norm(config.norm_type, config.d_model)
        self.ff = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask, self_rel_bias=None, cross_rel_bias=None):
        # Self Attention
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask, self_rel_bias))
        
        # Cross Attention
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, enc_output, enc_output, src_mask, cross_rel_bias))
        
        # FF
        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x

# ==========================================
# 4. Main Transformer Model
# ==========================================

class TransformerNMT(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.src_embedding = nn.Embedding(config.vocab_size_src, config.d_model, padding_idx=config.pad_idx)
        self.tgt_embedding = nn.Embedding(config.vocab_size_tgt, config.d_model, padding_idx=config.pad_idx)
        
        # Positional Encoding Strategy
        if config.pos_embedding_type == "absolute":
            self.pos_encoder = AbsolutePositionalEncoding(config.d_model, config.max_seq_len)
            self.rel_pos = None
        else:
            self.pos_encoder = lambda x: x # No-op
            # Shared relative bias logic usually for self-attn
            self.rel_pos_bias_enc = RelativePositionBias(config.n_heads, max_distance=config.max_seq_len)
            self.rel_pos_bias_dec = RelativePositionBias(config.n_heads, max_distance=config.max_seq_len)
            
        self.dropout = nn.Dropout(config.dropout)
        
        # Encoder & Decoder Stacks
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
        self.norm_enc = get_norm(config.norm_type, config.d_model)
        self.norm_dec = get_norm(config.norm_type, config.d_model)
        self.generator = nn.Linear(config.d_model, config.vocab_size_tgt)

    def make_src_mask(self, src):
        # src: [batch, len] -> mask: [batch, 1, 1, len]
        mask = (src != self.config.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt):
        # tgt: [batch, len]
        pad_mask = (tgt != self.config.pad_idx).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=tgt.device)).bool()
        
        mask = pad_mask & nopeak_mask
        return mask

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_encoder(self.src_embedding(src)))
        
        rel_bias = None
        if self.config.pos_embedding_type == "relative":
            rel_bias = self.rel_pos_bias_enc(src.size(1), src.size(1))
            
        for layer in self.encoder_layers:
            x = layer(x, src_mask, rel_pos_bias=rel_bias)
        return self.norm_enc(x)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.dropout(self.pos_encoder(self.tgt_embedding(tgt)))
        
        self_rel_bias = None
        if self.config.pos_embedding_type == "relative":
            self_rel_bias = self.rel_pos_bias_dec(tgt.size(1), tgt.size(1))
            
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask, self_rel_bias=self_rel_bias)
        return self.norm_dec(x)

    # === [Key Fix] forward now accepts mask parameter ===
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        return self.generator(dec_output)