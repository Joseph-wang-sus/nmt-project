import torch
import torch.nn.functional as F

def greedy_decode(model, src_ids, src_lengths, max_len, bos_id, eos_id):
    """Greedy decoding implementation."""
    model.eval()
    with torch.no_grad():
        batch_size = src_ids.size(0)
        enc_outputs, hidden = model.encoder(src_ids, src_lengths)
        pad_idx = getattr(model.config, "pad_idx_src", getattr(model.config, "pad_idx", 1))
        mask = (src_ids != pad_idx)
        
        input_step = torch.tensor([[bos_id]] * batch_size).to(src_ids.device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool).to(src_ids.device)
        decoded_ids = torch.zeros(batch_size, max_len, dtype=torch.long).to(src_ids.device)
        
        for t in range(max_len):
            output, hidden, _ = model.decoder(input_step, hidden, enc_outputs, mask)
            top1 = output.argmax(1) # (b,)
            
            decoded_ids[:, t] = top1
            input_step = top1.unsqueeze(1)
            
            # Update finished status
            finished |= (top1 == eos_id)
            if finished.all():
                break
                
        return decoded_ids

class BeamHypothesis:
    def __init__(self, sequence, score, hidden):
        self.sequence = sequence # List of token ids
        self.score = score       # Log probability
        self.hidden = hidden     # Decoder hidden state

def beam_search_decode(model, src_ids, src_lengths, max_len, bos_id, eos_id, beam_size=3):
    """
    Simple Beam Search for batch_size=1 (simplifies implementation for course level).
    If you need batch beam search, it gets much more complex.
    """
    model.eval()
    if src_ids.size(0) > 1:
        raise ValueError("Beam search currently supports batch_size=1 only.")

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(src_ids, src_lengths)
        pad_idx = getattr(model.config, "pad_idx_src", getattr(model.config, "pad_idx", 1))
        mask = (src_ids != pad_idx)
        
        # Initialize beam with Start Token
        # hidden is (layers, 1, h)
        init_hyp = BeamHypothesis(sequence=[bos_id], score=0.0, hidden=hidden)
        beam = [init_hyp]
        
        finished_beams = []
        
        for _ in range(max_len):
            new_beam = []
            for hyp in beam:
                # If hypothesis is finished, keep it
                if hyp.sequence[-1] == eos_id:
                    finished_beams.append(hyp)
                    continue
                
                # Prepare input
                input_step = torch.tensor([[hyp.sequence[-1]]]).to(src_ids.device)
                
                # Decode step
                output, new_hidden, _ = model.decoder(input_step, hyp.hidden, enc_outputs, mask)
                # output: (1, vocab)
                
                # Get top-k probabilities
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                # Expand beam
                for i in range(beam_size):
                    token_id = topk_ids[0, i].item()
                    score = topk_probs[0, i].item()
                    
                    new_seq = hyp.sequence + [token_id]
                    new_score = hyp.score + score
                    
                    new_beam.append(BeamHypothesis(new_seq, new_score, new_hidden))
            
            # Sort and Prune
            # Sort by score descending
            new_beam.sort(key=lambda x: x.score, reverse=True)
            beam = new_beam[:beam_size]
            
            # If all current beams are finished or worse than finished ones, stop
            if len(finished_beams) >= beam_size:
                break
        
        # Add remaining beams to finished
        finished_beams.extend(beam)
        finished_beams.sort(key=lambda x: x.score, reverse=True)
        
        # Return best sequence (excluding BOS)
        best_seq = finished_beams[0].sequence[1:] 
        return torch.tensor([best_seq], dtype=torch.long)