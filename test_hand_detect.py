import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab
from lib.utils.seed import set_all_random_seed
from lib.utils.logging import change_filehandler_mode_to_write

@hydra.main(version_base=None, config_path="conf", config_name="config_recognizer")
def main(cfg: DictConfig):
    root_logger = logging.getLogger()
    change_filehandler_mode_to_write(root_logger)

    logging.info(OmegaConf.to_yaml(cfg))

    set_all_random_seed(cfg.seed)

    # ===== Begin Configurations ===== #
    # == Train == #
    num_workers = cfg.train.num_workers
    test_dataset_partial = instantiate(cfg.test_dataset, _partial_=True)
    test_dataset_collate_fn = instantiate(cfg.test_collate_fn, _partial_=True)

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

    test_batch_size = cfg.get('test_batch_size', 32)
    dataset_test = test_dataset_partial(vocab=vocab_map)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset_collate_fn,
    )

    recognizer = recognizer_partial(
        encoder_out_dim=len(char_list),
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(cfg.recognizer.ckpt_path))
    recognizer.eval()

    df_fs = pd.read_csv('data/CustomChicagoFSWildFinal.csv')
    correct_cnt = 0
    total_cnt = 0
    one_hand_video_names = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            rlh_seg = batch['rlh_seg'].cuda()
            poses = batch['poses'].cuda()
            target_ids = batch['target_ids'].cuda()
            frame_idx = batch['frame_idx'].cuda()
            video_names = batch['video_name']

            B, L_poses = poses.shape[:2]
            _, L_tg = target_ids.shape[:2]

            recognizer_out = recognizer.generate(poses, rlh_seg, frame_idx, bos_token_id=len(char_list), eos_token_id=0)
            cross_attn_tokens = recognizer_out['cross_attns']
            for b in range(B):
                if len(rlh_seg[b].unique()) < 2:
                    one_hand_video_name = video_names[b]
                    one_hand_video_names.append(one_hand_video_name)
                    continue
                cats = []
                for l in range(3): # 3: the number of layers
                    cats.append(cross_attn_tokens[l][b][-1].unsqueeze(0))  # decoder layer, batch
                cats = torch.cat(cats)
                cats = cats.mean(0) # |W|, T

                seg = rlh_seg[b].cpu()
                hand_ids = seg.unique()

                hand_scores = []

                for h in hand_ids:
                    mask = (seg == h)
                    if mask.sum() == 0:
                        hand_scores.append(torch.tensor(-1e9, device=cats.device))
                        continue

                    # (|W|,) → scalar
                    score = cats[:, mask].sum()
                    hand_scores.append(score)

                hand_scores = torch.stack(hand_scores)  # (num_hands,)
                max_hand_idx = hand_scores.argmax()
                majority_hand = int(hand_ids[max_hand_idx].item())

                person_id = majority_hand // 2 + 1
                majority_hand = majority_hand % 2

                video_name = video_names[b]

                gt_hand_type = df_fs[df_fs['filename'].str.contains(video_name)]['GT_hand_type'].iloc[0]
                gt_person_id = df_fs[df_fs['filename'].str.contains(video_name)]['GT_person_id'].iloc[0]

                # Change condition for incrementing correct_cnt
                is_hand_type_correct = False
                if gt_hand_type in ["R", "L"]:
                    if gt_hand_type == "R":
                        gt_hand_type_val = 0
                    elif gt_hand_type == "L":
                        gt_hand_type_val = 1

                    if gt_hand_type_val == majority_hand:
                        is_hand_type_correct = True
                    else:
                        logging.info(f"Hand type mismatch: GT={gt_hand_type_val}, Predicted={majority_hand}, Video={video_name}")

                is_person_id_correct = False
                if pd.isna(gt_person_id):
                    is_person_id_correct = True # If gt_person_id is NaN, person_id match is not considered
                else:
                    if gt_person_id == person_id:
                        is_person_id_correct = True
                    else:
                        logging.info(f"Person ID mismatch: GT={gt_person_id}, Predicted={person_id}, Video={video_name}")

                # Increment correct_cnt only if both conditions are true
                if is_hand_type_correct and is_person_id_correct:
                    correct_cnt += 1
                else:
                    pass

                if gt_hand_type in ["R", "L"]:
                    total_cnt += 1
                    print(f"{correct_cnt / total_cnt * 100:.2f}%", end='\r')

    logging.info(f"Acc: {correct_cnt / total_cnt * 100:.2f}%, N: {total_cnt}")

    with open('data/one_hand_video_names.txt', 'w') as f:
        for one_hand_video_name in one_hand_video_names:
            f.write(f'{one_hand_video_name}\n')


if __name__ == "__main__":
    main()