# from  https://github.com/chevalierNoir/asl-iter-attn/blob/75208c5bebc16ed9c80a757d2586562adc30ef49/lev.py#L62


from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F

# subs = np.zeros((26,26))
# subs = np.zeros((51+26+10, 51+26+10))

def measure_hand_focus(attn_matrix, split_idx, normalize_rows=True):
    L, T = attn_matrix.shape
    device = attn_matrix.device

    right_mask = torch.zeros(T, device=device)
    right_mask[:split_idx] = 1.0
    left_mask = 1.0 - right_mask

    A = attn_matrix.clamp_min(0)
    if normalize_rows:
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

    right_ratio = (A * right_mask).sum(dim=1)
    left_ratio  = (A * left_mask).sum(dim=1)
    diff_rl     = right_ratio - left_ratio

    per_letter = {
        "right_ratio": right_ratio,
        "left_ratio":  left_ratio,
        "diff_rl":     diff_rl
    }
    overall = {
        "right_mean": right_ratio.mean().item(),
        "left_mean":  left_ratio.mean().item(),
        "diff_mean":  diff_rl.mean().item()
    }
    return per_letter, overall

def measure_monotonicity(attn_matrix, split_idx):
    attn_matrix = attn_matrix[:, :split_idx]
    L, T = attn_matrix.shape

    frame_indices = torch.arange(T, device=attn_matrix.device)
    centers = (attn_matrix * frame_indices).sum(dim=1) / (attn_matrix.sum(dim=1) + 1e-8)

    diffs = centers[1:] - centers[:-1]

    monotonic_ratio = (diffs > 0).float().mean().item()

    return monotonic_ratio, centers

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """
    Computes Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t
    s: source, t: target
    costs: a tuple or a list with three integers (d, i, s)
           where d defines the costs for a deletion
                 i defines the costs for an insertion and
                 s defines the costs for a substitution
    return:
    H, S, D, I: correct chars, number of substitutions, number of deletions, number of insertions
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    H, D, S, I = 0, 0, 0, 0
    for row in range(1, rows):
        dist[row][0] = row * deletes
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost)
    row, col = rows-1, cols-1
    while row != 0 or col != 0:
        if row == 0:
            I += col
            col = 0
        elif col == 0:
            D += row
            row = 0
        elif dist[row][col] == dist[row-1][col] + deletes:
            D += 1
            row = row-1
        elif dist[row][col] == dist[row][col-1] + inserts:
            I += 1
            col = col-1
        elif dist[row][col] == dist[row-1][col-1] + substitutes:
            S += 1
            row, col = row-1, col-1

        else:
            H += 1
            row, col = row-1, col-1
    D, I = I, D
    return H, D, S, I

def compute_simple_acc(preds, labels):
    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) * 100

def compute_acc(preds, labels, costs=(7, 7, 10)):
    # cost according to HTK: http://www.ee.columbia.edu/~dpwe/LabROSA/doc/HTKBook21/node142.html

    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')
    Ns, Ds, Ss, Is = 0, 0, 0, 0
    for i, _ in enumerate(preds):
        H, D, S, I = iterative_levenshtein(preds[i], labels[i], costs)
        # print(H, D, S, I)
        Ns += len(labels[i])
        Ds += D
        Ss += S
        Is += I
    try:
        acc = 100*(Ns-Ds-Ss-Is)/Ns
    except ZeroDivisionError as err:
        raise ZeroDivisionError('Empty labels')
    return acc, Ds, Ss, Is, Ns

def compute_acc_frame(preds, labels):
    summation = 0
    cnt = 0
    for pred, label in zip(preds, labels):
        summation += np.sum(pred==label)
        cnt += len(label[0])
    acc = summation/cnt*100
    return acc

def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='none', logits_lm =None ):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # ==========================================================================================================
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # ==========================================================================================================
        probs = log_probs[:input_length, i].exp()
        # ==========================================================================================================
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        logits_lm = F.softmax(logits_lm, dim=-1)

        lm_alpha = log_probs.new_zeros((target_length * 2 + 1,),dtype=logits_lm.dtype).float()
        lm_alpha[1::2] = torch.diagonal(logits_lm[:,targets[0]], 0)

        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])

        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += (alpha[:-1] + lm_alpha[1:])
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
            # if logits_lm != None:
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count