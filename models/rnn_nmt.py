import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass
from models.attention import DotAttention, GeneralAttention, AdditiveAttention

@dataclass
class RNNConfig:
    vocab_size_src: int
    vocab_size_tgt: int
    embed_size: int = 256
    hidden_size: int = 512
    rnn_type: str = "gru"        # "gru" or "lstm"
    enc_layers: int = 2
    dec_layers: int = 2
    dropout: float = 0.3
    attention_type: str = "dot"  # "dot", "general", "additive"
    pad_idx_src: int = 1
    pad_idx_tgt: int = 1

class RNNEncoder(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.rnn_type = config.rnn_type
        self.embedding = nn.Embedding(config.vocab_size_src, config.embed_size, padding_idx=config.pad_idx_src)
        
        rnn_cls = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.enc_layers,
            batch_first=True,
            dropout=config.dropout if config.enc_layers > 1 else 0
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src_ids, src_lengths):
        # src_ids: (b, seq_len)
        embedded = self.dropout(self.embedding(src_ids))
        
        # Pack sequence for efficient RNN processing
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        output, hidden = self.rnn(packed)
        
        # Unpack
        # outputs: (b, seq_len, hidden)
        outputs, _ = pad_packed_sequence(output, batch_first=True)
        
        # hidden: (n_layers, b, hidden) - for GRU
        # hidden: ((n_layers, b, hidden), (n_layers, b, hidden)) - for LSTM
        return outputs, hidden

class RNNDecoder(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.rnn_type = config.rnn_type
        self.embedding = nn.Embedding(config.vocab_size_tgt, config.embed_size, padding_idx=config.pad_idx_tgt)
        
        # Attention Mechanism Selection
        if config.attention_type == "dot":
            self.attention = DotAttention()
        elif config.attention_type == "general":
            self.attention = GeneralAttention(config.hidden_size)
        elif config.attention_type == "additive":
            self.attention = AdditiveAttention(config.hidden_size)
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")
            
        rnn_cls = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
        
        # Input to decoder RNN is: Embedding + Context Vector
        self.rnn = rnn_cls(
            input_size=config.embed_size + config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.dec_layers,
            batch_first=True,
            dropout=config.dropout if config.dec_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size_tgt)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_step, hidden, enc_outputs, mask):
        # input_step: (b, 1) - One token
        # hidden: previous hidden state
        # enc_outputs: (b, src_len, hidden)
        
        embedded = self.dropout(self.embedding(input_step)) # (b, 1, embed)
        
        # Calculate Attention
        # Use the top layer of hidden state for attention query
        # hidden is (layers, b, hidden)
        if self.rnn_type == "lstm":
            # (h, c) tuple. Query comes from h.
            query = hidden[0][-1] 
        else:
            # hidden tensor. Query is last layer
            query = hidden[-1]
            
        context, attn_weights = self.attention(query, enc_outputs, mask)
        # context: (b, hidden) -> unsqueeze to (b, 1, hidden)
        context = context.unsqueeze(1)
        
        # Input Feeding: Concatenate embedding and context
        rnn_input = torch.cat((embedded, context), dim=2) # (b, 1, embed + hidden)
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output: (b, 1, hidden)
        
        prediction = self.fc_out(output.squeeze(1)) # (b, vocab)
        
        return prediction, hidden, attn_weights

class RNNNMT(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config
        self.encoder = RNNEncoder(config)
        self.decoder = RNNDecoder(config)

    def forward(self, src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=0.5):
        # src_ids: (b, src_len)
        # tgt_ids: (b, tgt_len) - Includes <bos> and <eos>
        
        batch_size = src_ids.shape[0]
        tgt_len = tgt_ids.shape[1]
        vocab_size = self.config.vocab_size_tgt
        
        # Run Encoder
        # enc_outputs: (b, src_len, hidden)
        # hidden: (layers, b, hidden)
        enc_outputs, hidden = self.encoder(src_ids, src_lengths)
        
        # Create mask for attention (True where NOT pad)
        # Assuming src_ids has 0 or 1 as pad.
        mask = (src_ids != self.config.pad_idx_src)
        
        # Prepare outputs tensor
        # We will store logits here.
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src_ids.device)
        
        # First input is <bos> token
        input_step = tgt_ids[:, 0].unsqueeze(1) # (b, 1)
        
        # Loop through target length (skip the first token which is BOS in the output buffer, 
        # usually we predict from t=1 to end)
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_step, hidden, enc_outputs, mask)
            
            outputs[:, t, :] = output
            
            # Decision: Teacher Forcing vs Free Running
            top1 = output.argmax(1) # Model prediction
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                next_input = tgt_ids[:, t] # Ground truth
            else:
                next_input = top1 # Model prediction
            
            input_step = next_input.unsqueeze(1)
            
        return outputs