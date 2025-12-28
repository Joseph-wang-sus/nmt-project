import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (batch, hidden_dim) - Decoder hidden state
            keys:  (batch, src_len, hidden_dim) - Encoder outputs
            mask:  (batch, src_len) - Boolean mask (True where valid, False where padding)
        
        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, src_len)
        """
        # Calculate alignment scores
        scores = self.score(query, keys)  # (batch, src_len)
        
        # Apply mask (set padding positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, src_len)
        
        # Weighted sum of keys (encoder outputs)
        # (batch, 1, src_len) @ (batch, src_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attn_weights

    def score(self, query, keys):
        raise NotImplementedError

class DotAttention(AttentionBase):
    """score(h_t, h_s) = h_t^T . h_s"""
    def score(self, query, keys):
        # query: (b, h), keys: (b, l, h)
        # Expand query to (b, h, 1) for batched matmul or just use unsqueeze
        # (b, l, h) @ (b, h, 1) -> (b, l, 1)
        return torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

class GeneralAttention(AttentionBase):
    """score(h_t, h_s) = h_t^T . W . h_s"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def score(self, query, keys):
        # W(keys): (b, l, h)
        x = self.W(keys)
        # (b, l, h) @ (b, h, 1) -> (b, l, 1)
        return torch.bmm(x, query.unsqueeze(2)).squeeze(2)

class AdditiveAttention(AttentionBase):
    """score(h_t, h_s) = v^T . tanh(W1 h_t + W2 h_s)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, query, keys):
        # query: (b, h) -> (b, 1, h)
        q = self.W1(query).unsqueeze(1)
        # keys: (b, l, h)
        k = self.W2(keys)
        
        # broadcast add: (b, 1, h) + (b, l, h) -> (b, l, h)
        energy = torch.tanh(q + k)
        
        # v: (b, l, h) -> (b, l, 1)
        scores = self.v(energy).squeeze(2)
        return scores