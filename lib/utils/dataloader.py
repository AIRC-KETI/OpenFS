import numpy as np
import torch
import constants

def paired_collate_recog_fn(insts):
    poses = pad_collate_fn([inst['poses'] for inst in insts], constants.MINUS_TWO_VALUE)
    rlh_seg = pad_collate_fn([inst['rlh_seg'] for inst in insts], 32)
    frame_idx = pad_collate_fn([inst['frame_idx'] for inst in insts], 1024)
    target_ids = pad_collate_fn([inst['target_ids'] for inst in insts], constants.MINUS_ONE_HUNDRED_VALUE)

    batch = {
        'poses': poses[0],
        'rlh_seg': rlh_seg[0],
        'frame_idx': frame_idx[0],
        'target_ids': target_ids[0],
    }
    if 'video_name' in insts[0]:
        batch['video_name'] = [inst['video_name'] for inst in insts]
    if 'word' in insts[0]:
        batch['word'] = [inst['word'] for inst in insts]
    return batch

def paired_collate_word_dataset_fn(insts):
    frame_idx_gen = pad_collate_fn([inst['frame_idx_gen'] for inst in insts], 1024)
    frame_label_gen = pad_collate_fn([inst['frame_label_gen'] for inst in insts], constants.MINUS_ONE_HUNDRED_VALUE)
    target_ids_gen = pad_collate_fn([inst['target_ids_gen'] for inst in insts], constants.MINUS_ONE_HUNDRED_VALUE)

    batch = {
        'frame_idx_gen': frame_idx_gen[0],
        'frame_label_gen': frame_label_gen[0],
        'target_ids_gen': target_ids_gen[0],
    }
    if 'word' in insts[0]:
        batch['word'] = [inst['word'] for inst in insts]
    return batch

def paired_collate_gen_fn(insts):
    input_ids = pad_collate_fn([inst['input_ids'] for inst in insts], constants.MINUS_ONE_HUNDRED_VALUE)
    poses = pad_collate_fn([inst['poses'] for inst in insts], constants.MINUS_TWO_VALUE)
    frame_idx = pad_collate_fn([inst['frame_idx'] for inst in insts], 1024)

    batch = {
        'input_ids': input_ids[0],
        'poses': poses[0],
        'frame_idx': frame_idx[0],
    }
    if 'frame_label' in insts[0]:
        frame_label = pad_collate_fn([inst['frame_label'] for inst in insts], constants.MINUS_ONE_HUNDRED_VALUE)
        batch['frame_label'] = frame_label[0]
    if 'video_name' in insts[0]:
        batch['video_name'] = [inst['video_name'] for inst in insts]
    if 'word' in insts[0]:
        batch['word'] = [inst['word'] for inst in insts]
    return batch

def pad_collate_fn(batch, pad_value, padding_side='right'):
    # Determine ndim and feature shape from first non-empty sample
    sample_ndim, feature_shape = None, None
    for sample in batch:
        if len(sample) > 0:
            arr = np.array(sample)
            sample_ndim = arr.ndim
            feature_shape = arr.shape[1:] if arr.ndim > 1 else ()
            break

    if sample_ndim is None:
        # All samples are empty → return shape (B, 1, 21, 3) filled with pad_value
        B = len(batch)
        padded_shape = (B, 1, *batch[0].shape[1:])  # or (B, 1, *feature_shape) if general
        # 21, 3 for hand poses
        return (
            torch.full(padded_shape, pad_value, dtype=torch.float32),
            torch.zeros((B, 1), dtype=torch.long)
        )

    max_len = max(len(sample) for sample in batch)
    padded_seqs = []
    position_indices = []

    for sample in batch:
        if len(sample) == 0:
            # Fill with pad_value to shape (max_len, ...)
            if sample_ndim == 1:
                padded = np.full((max_len,), pad_value)
                pos = [0] * max_len
            elif sample_ndim == 2:
                padded = np.full((max_len, feature_shape[0]), pad_value)
                pos = [0] * max_len
            elif sample_ndim == 3:
                padded = np.full((max_len, feature_shape[0], feature_shape[1]), pad_value)
                pos = [0] * max_len
        else:
            arr = np.array(sample)
            pad_len = max_len - len(arr)

            if padding_side == 'right':
                pad_before, pad_after = 0, pad_len
            else:  # left padding
                pad_before, pad_after = pad_len, 0

            if sample_ndim == 1:
                padded = np.pad(arr, (pad_before, pad_after), constant_values=pad_value)
                pos = [i + 1 if token != pad_value else 0 for i, token in enumerate(padded)]
            elif sample_ndim == 2:
                padded = np.pad(arr, ((pad_before, pad_after), (0, 0)), constant_values=pad_value)
                pos = [i + 1 if (frame != pad_value).all() else 0 for i, frame in enumerate(padded)]
            elif sample_ndim == 3:
                padded = np.pad(arr, ((pad_before, pad_after), (0, 0), (0, 0)), constant_values=pad_value)
                pos = [i + 1 if (frame != pad_value).all() else 0 for i, frame in enumerate(padded)]

        padded_seqs.append(torch.tensor(padded))
        position_indices.append(torch.tensor(pos))

    return torch.stack(padded_seqs), torch.stack(position_indices)