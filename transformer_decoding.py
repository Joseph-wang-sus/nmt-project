import torch
import torch.nn.functional as F

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    # [New] If it is a DataParallel object, extract the internal original model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    """Transformer Greedy Decoding"""
    # 1. Encode
    memory = model.encode(src, src_mask)
    # 2. Init Decoder Input
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_size=5, alpha=0.6):
    """Transformer Beam Search Decoding"""
    # Ensure batch size is 1 for this implementation
    if src.size(0) != 1:
        raise ValueError("Beam search currently supports batch_size=1 only.")
        
    # 1. Encode
    memory = model.encode(src, src_mask)
    # Repeat memory for beam size: [beam, seq_len, d_model]
    memory = memory.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    # 2. Init
    ys = torch.ones(beam_size, 1).fill_(start_symbol).type(torch.long).to(device)
    scores = torch.zeros(beam_size).to(device)
    scores[1:] = -1e9 
    
    finished_seqs = []
    finished_scores = []

    for i in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        log_probs = F.log_softmax(model.generator(out[:, -1]), dim=-1)
        
        # Add current scores
        # scores: [beam, 1], log_probs: [beam, vocab]
        next_scores = scores.unsqueeze(1) + log_probs
        next_scores_flat = next_scores.view(-1)
        
        # Top-k
        topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size)
        
        vocab_size = log_probs.size(1)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size
        
        # Reconstruct
        # Prepare for next step
        ys_new = torch.zeros(beam_size, ys.size(1)+1, dtype=torch.long).to(device)
        next_beam_scores = []
        active_beams = 0
        
        for k in range(beam_size):
            idx = beam_indices[k]
            token = token_indices[k]
            score = topk_scores[k]
            prev_seq = ys[idx]
            
            if token.item() == end_symbol:
                penalty = (len(prev_seq) + 1) ** alpha
                finished_seqs.append(torch.cat([prev_seq, torch.tensor([token]).to(device)]))
                finished_scores.append(score / penalty)
                next_beam_scores.append(-1e9) # Mark as finished
                # Fill dummy for tensor shape consistency
                ys_new[k] = torch.cat([prev_seq, torch.tensor([end_symbol]).to(device)]) 
            else:
                ys_new[k] = torch.cat([prev_seq, torch.tensor([token]).to(device)])
                next_beam_scores.append(score)
                active_beams += 1
        
        ys = ys_new
        scores = torch.tensor(next_beam_scores).to(device)
        
        if active_beams == 0:
            break
            
    # Select best
    if len(finished_seqs) == 0:
        return ys[0].unsqueeze(0) 
    
    best_idx = finished_scores.index(max(finished_scores))
    return finished_seqs[best_idx].unsqueeze(0)