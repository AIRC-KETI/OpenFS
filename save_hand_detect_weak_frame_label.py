import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import tqdm
import re

import torch
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab
from lib.utils.seed import set_all_random_seed

@hydra.main(version_base=None, config_path="conf", config_name="config_recognizer")
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))

    set_all_random_seed(cfg.seed)

    # ===== Begin Configurations ===== #
    # == Train == #
    num_workers = cfg.train.num_workers

    # == Dataset == #
    dataset_partial = instantiate(cfg.dataset, _partial_=True)
    dataset_collate_fn = instantiate(cfg.collate_fn, _partial_=True)

    # == Model == #
    recognizer_partial = instantiate(cfg.recognizer, _partial_=True)

    # == Decode == #
    chars = cfg.decode.chars

    # ===== End Configurations ===== #

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f'device {device}')

    vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)

    logging.info(vocab_map)
    logging.info(inv_vocab_map)
    logging.info(char_list)

    dataset = dataset_partial(
        vocab=vocab_map, return_video_name=True, do_augm=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_collate_fn,
    )

    recognizer = recognizer_partial(
        encoder_out_dim=len(char_list),
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(cfg.recognizer.ckpt_path))
    recognizer.eval()

    majority_hands = []
    frame_labels = []

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            poses = batch['poses'].cuda()
            rlh_seg = batch['rlh_seg'].cuda()
            frame_idx = batch['frame_idx'].cuda()
            target_ids = batch['target_ids'].cuda()
            video_name = batch['video_name'][0]
            B, L_poses = poses.shape[:2]

            recognizer_out = recognizer(
                poses,
                rlh_seg,
                frame_idx,
                target_ids[:, :-1]
            )
            cross_attn_tokens = recognizer_out['cross_attns']

            for b in range(B):
                cats = []
                for l in range(3): # 3: the number of layers
                    cats.append(cross_attn_tokens[l][b].unsqueeze(0))  # decoder layer, First batch
                cats = torch.cat(cats)
                cats = cats.mean(0)

                key_frame = cats.argmax(dim=1)
                key_hand_list = rlh_seg[:, key_frame]
                values, counts = torch.unique(key_hand_list, return_counts=True)
                majority_hand = values[torch.argmax(counts)].item()

                cats_masked = cats[:, (rlh_seg==majority_hand).squeeze(0)]
                T, F = cats_masked.shape # the number of text token, frame
                if F < 5 or T > F:
                    majority_hands.append(None)
                    frame_labels.append(None)
                    continue

                labels = torch.full((T, F), -1, dtype=torch.long)
                target_id = target_ids[b]

                for t in range(T-1):
                    label = target_id[t+1]
                    cat_t = cats_masked[t]
                    val = cat_t.topk(4).values[1:].mean()
                    low = val*0.5
                    mask = cat_t >= low
                    labels[t, mask] = label.cpu()

                final_labels = torch.full((F,), -1, dtype=torch.long)
                for f in range(F):
                    vals = labels[:, f]
                    unique_vals = torch.unique(vals[vals != -1])
                    if len(unique_vals) == 1:
                        final_labels[f] = unique_vals[0]
                    else:
                        final_labels[f] = -1

                majority_hands.append(majority_hand)
                frame_labels.append(final_labels.numpy())

    data = np.load(cfg.dataset.data_path, allow_pickle=True)
    data = dict(data)
    data['signing_hand'] = majority_hands
    data['frame_label'] = frame_labels

    np.savez(cfg.dataset.data_path.replace(".npz", "_proc.npz"), **data)


if __name__ == "__main__":
    main()