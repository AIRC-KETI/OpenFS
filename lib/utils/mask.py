import torch

def get_batch_attention_mask(seq, pad_value):
    if seq.dim() == 4:
        non_pad_mask = (seq != pad_value).any(dim=-1).any(dim=-1).float()
    elif seq.dim() == 3:
        non_pad_mask = (seq != pad_value).any(dim=-1).float()
    elif seq.dim() == 2:
        non_pad_mask = (seq != pad_value).float()
    else:
        raise ValueError(f"Unsupported tensor dimension: {seq.dim()}")

    mask = torch.where(
        non_pad_mask == 1,
        torch.tensor(0.0, device=seq.device),
        torch.tensor(float('-inf'), device=seq.device)
    )
    return mask
