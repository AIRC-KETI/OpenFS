import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_ce(logits, labels, masks, bgw=1.0):
    if bgw < 1.0:
        weight = torch.ones(logits.shape[-1]).to(logits.device)
        weight[0] = bgw     # set small weight to pseudo background frames
        logits = logits[masks]     # (n, cls)
        labels = labels[masks]
        return F.cross_entropy(logits, labels, weight=weight)

    ce = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    return torch.mean(ce[masks])

def get_mse_loss_with_ignore(pred, target, valid_mask):
    # Apply mask
    pred_valid = pred[valid_mask]
    tgt_valid = target[valid_mask]

    # Compute L1 loss
    return F.mse_loss(pred_valid, tgt_valid)

def get_attn_ent_loss(cross_attns, rlh_seg, target_ids):
    avg_attn = cross_attns.mean(dim=1)
    K = rlh_seg.max().item() + 1
    seg_onehot = F.one_hot(rlh_seg, num_classes=K).float()
    S = torch.einsum('bts, bsk -> btk', avg_attn, seg_onehot)
    ent = -(S * (S + 1e-12).log()) #.sum(dim=2)
    ent_masked = ent*(target_ids[:, 1:] != -100).unsqueeze(-1)
    ent_masked = ent_masked[ent_masked!=0]
    ent_loss = ent_masked.mean()
    return ent_loss

def get_soft_monotonicity_loss(cross_attn, rlh_seg, target_ids, threshold=1e-10):
    """
    cross_attn: Tensor of shape (B, L, T_out, T_in)
    threshold: values below this will be clamped to 0
    """
    B, L, T_out, T_in = cross_attn.shape
    total_loss = 0.0

    seg_mask = (rlh_seg != 32)
    tgt_mask = target_ids[:, 1:] != -100
    combined_mask = tgt_mask.unsqueeze(-1) & seg_mask.unsqueeze(1)  # shape: (B, T, S)

    for l in range(L):
        attn = cross_attn[:, l]  # shape: (B, T_out, T_in)
        C = torch.cumsum(attn, dim=2)   # cumulative attention over input tokens
        diff = C[:, 1:, :] - C[:, :-1, :]  # (B, T_out - 1, T_in)

        # Soft monotonicity violation
        violation = F.relu(diff)
        masked_violation = violation * combined_mask[:, 1:]
        valid_count = combined_mask.sum()
        if valid_count > 0:
            total_loss += masked_violation.sum() / valid_count

    return total_loss