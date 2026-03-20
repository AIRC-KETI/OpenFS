import math
import torch

def sinusoidal_positional_encoding(seq_len, dim, device='cuda'):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def positional_encoding_from_idx(input_idx, dim, device='cuda'):
    """
    Generate sinusoidal positional encodings based on explicit frame indices.

    Args:
        input_idx (array-like or Tensor): Tensor of shape (B, T) containing frame indices.
        dim (int): Dimensionality of the positional encoding.
        device (torch.device, optional): Device on which to create the output Tensor.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (B, T, dim).
    """
    if not torch.is_tensor(input_idx):
        input_idx = torch.tensor(input_idx, dtype=torch.float32)

    input_idx = input_idx.to(device)

    B, T = input_idx.shape
    pe = torch.zeros(B, T, dim, device=device)

    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )

    angles = input_idx.unsqueeze(-1) * div_term  # shape: (B, T, dim/2)
    pe[:, :, 0::2] = torch.sin(angles)
    pe[:, :, 1::2] = torch.cos(angles)

    return pe

def get_sinusoidal_time_embedding(time_seg, d_model, padding_value=-100):
    """
    time_seg: (B, L) int64 tensor with padding (e.g., -100)
    returns: (B, L, d_model) float tensor with 0s at padding positions
    """
    device = time_seg.device
    B, L = time_seg.shape

    # Mask and clamp
    time_seg_clamped = time_seg.clone().float()
    time_seg_clamped[time_seg == padding_value] = 0  # Temporarily set to 0

    time_seg_clamped = time_seg_clamped.unsqueeze(-1)  # (B, L, 1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe = torch.zeros(B, L, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(time_seg_clamped * div_term)
    pe[:, :, 1::2] = torch.cos(time_seg_clamped * div_term)

    # Zero out padded positions
    mask = (time_seg != padding_value).unsqueeze(-1)  # (B, L, 1)
    pe = pe * mask

    return pe

def build_frame_idx_ext_torch(frame_idx, word_label, max_val):
    B, T = frame_idx.shape
    L_w = word_label.shape[1]
    device = frame_idx.device
    dtype = frame_idx.dtype

    pe = torch.arange(L_w, device=device, dtype=dtype).unsqueeze(0).expand(B, L_w)      # (B, L_w)
    tail = frame_idx + L_w                                                               # (B, T)
    frame_idx_ext = torch.cat([pe, tail], dim=1)                                         # (B, L_w+T)

    mask = (word_label == 33)                                                            # (B, L_w)
    frame_idx_ext[:, :L_w][mask] = max_val

    frame_idx_ext.clamp_(min=0, max=max_val)
    return frame_idx_ext

def build_frame_idx_ext_torch_feats(frame_idx, word_feats, max_val):
    B, T = frame_idx.shape
    L_w = word_feats.shape[1]
    device = frame_idx.device
    dtype = frame_idx.dtype

    pe = torch.arange(L_w, device=device, dtype=dtype).unsqueeze(0).expand(B, L_w)      # (B, L_w)
    tail = frame_idx + L_w                                                               # (B, T)
    frame_idx_ext = torch.cat([pe, tail], dim=1)                                         # (B, L_w+T)

    frame_idx_ext.clamp_(min=0, max=max_val)
    return frame_idx_ext