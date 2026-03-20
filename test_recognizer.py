import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import time
import logging
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab, invert_to_chars
from lib.utils.eval import compute_acc
from lib.utils.time import seconds_to_hhmmss
from lib.utils.seed import set_all_random_seed
from lib.utils.logging import change_filehandler_mode_to_write
from lib.utils.size import count_parameters

@hydra.main(version_base=None, config_path="conf", config_name="config_recognizer")
def main(cfg: DictConfig):
    root_logger = logging.getLogger()
    change_filehandler_mode_to_write(root_logger)

    logging.info(OmegaConf.to_yaml(cfg))

    set_all_random_seed(cfg.seed)

    # ===== Begin Configurations ===== #
    # == Train == #
    num_workers = cfg.train.num_workers

    # == Dataset == #
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
    logging.info(f'Model size: {count_parameters(recognizer)}')

    eval_start_time = time.time()
    preds = []
    gts = []

    preds_oov = []
    gts_oov = []

    preds_iv = []
    gts_iv = []

    preds_fair = []
    gts_fair = []

    total_letter_count = 0
    total_seq_count = 0

    with open('data/english_word/Chicago_train.txt', 'r') as f:
        iv_labels = f.read().splitlines()

    with open('data/posenet_signhand_miss.txt', 'r') as f:
        passing = f.read().splitlines()

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            rlh_seg = batch['rlh_seg'].cuda()
            poses = batch['poses'].cuda()
            target_ids = batch['target_ids'].cuda()
            frame_idx = batch['frame_idx'].cuda()
            if 'video_name' in batch:
                video_names = batch['video_name']
            else:
                video_names = None

            recognizer_out = recognizer.generate(poses, rlh_seg, frame_idx, bos_token_id=len(char_list), eos_token_id=0)
            output_ids = recognizer_out['output_ids']

            B = output_ids.shape[0]
            for b in range(B):
                current_pred = ''.join(invert_to_chars(output_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))
                gt_label = ''.join(invert_to_chars(target_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))

                preds.append(current_pred)
                gts.append(gt_label)
                _lev_acc, _, _, _, _ = compute_acc([current_pred], [gt_label])
                if gt_label in iv_labels:
                    preds_iv.append(current_pred)
                    gts_iv.append(gt_label)
                else:
                    preds_oov.append(current_pred)
                    gts_oov.append(gt_label)

                if video_names is not None:
                    video_name = video_names[b]
                else:
                    video_name = None

                if video_name is not None and video_name.split('/')[-1] not in passing:
                    preds_fair.append(current_pred)
                    gts_fair.append(gt_label)

                total_letter_count += len(current_pred)
                total_seq_count += (frame_idx[b] != 1024).sum().item()
                logging.info(f'pred: {current_pred} \t gt: {gt_label} \t acc: {_lev_acc:.2f} {video_name}  {i+1}/{len(test_dataloader)}  {seconds_to_hhmmss(time.time()-eval_start_time)}')

    total_time = time.time()-eval_start_time
    lev_acc, Ds, Ss, Is, Ns = compute_acc(preds, gts)
    logging.info(f'[Test] - Letter Acc: {lev_acc:.4f} - Deletion: {Ds} - Substitution: {Ss} - Insertion: {Is}, Num: {Ns}, Num S: {len(preds)}')

    if len(gts_oov) > 0:
        lev_acc, Ds, Ss, Is, Ns = compute_acc(preds_oov, gts_oov)
        logging.info(f'[Test oov] - Letter Acc: {lev_acc:.4f} - Deletion: {Ds} - Substitution: {Ss} - Insertion: {Is}, Num: {Ns}, Num S: {len(preds_oov)}')

    if len(gts_iv) > 0:
        lev_acc, Ds, Ss, Is, Ns = compute_acc(preds_iv, gts_iv)
        logging.info(f'[Test iv] - Letter Acc: {lev_acc:.4f} - Deletion: {Ds} - Substitution: {Ss} - Insertion: {Is}, Num: {Ns}, Num S: {len(preds_iv)}')

    if len(gts_fair) > 0:
        lev_acc, Ds, Ss, Is, Ns = compute_acc(preds_fair, gts_fair)
        logging.info(f'[Test fair] - Letter Acc: {lev_acc:.4f} - Deletion: {Ds} - Substitution: {Ss} - Insertion: {Is}, Num: {Ns}, Num S: {len(preds_fair)}')

    lps = total_letter_count / total_time
    fps = total_seq_count / total_time
    logging.info(f'LPS letters per second {lps:.2f}, total letter count: {total_letter_count}, total latency: {total_time:.2f}')
    logging.info(f'fps {fps:.2f}, total seq count: {total_seq_count}, total latency: {total_time:.2f}')
    troughput = len(gts)/total_time
    logging.info(f'Troughput {troughput:.2f}')

    word_acc = np.mean(np.array(preds) == np.array(gts))
    word_acc_oov = np.mean(np.array(preds_oov) == np.array(gts_oov))
    word_acc_iv = np.mean(np.array(preds_iv) == np.array(gts_iv))
    word_acc_fair = np.mean(np.array(preds_fair) == np.array(gts_fair))

    logging.info(f'Word Acc: {word_acc:.4f}')
    logging.info(f'Word Acc FAIR: {word_acc_fair:.4f}')
    logging.info(f'Word Acc IV: {word_acc_iv:.4f}')
    logging.info(f'Word Acc OOV: {word_acc_oov:.4f}')

if __name__ == "__main__":
    main()