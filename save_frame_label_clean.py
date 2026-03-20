import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import tqdm
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab
from lib.utils.seed import set_all_random_seed
from lib.utils.logging import change_filehandler_mode_to_write

@hydra.main(version_base=None, config_path="conf", config_name="config_frame_classifier")
def main(cfg: DictConfig):
    root_logger = logging.getLogger()
    change_filehandler_mode_to_write(root_logger)

    logging.info(OmegaConf.to_yaml(cfg))

    set_all_random_seed(cfg.seed)

    # ===== Begin Configurations ===== #
    # == Train == #
    num_workers = cfg.train.num_workers

    dataset_partial = instantiate(cfg.dataset, _partial_=True)
    dataset_collate_fn = instantiate(cfg.collate_fn, _partial_=True)

    # == Model == #
    frame_classifier_partial = instantiate(cfg.frame_classifier, _partial_=True)
    frame_classifier_ckpt_path = cfg.frame_classifier.ckpt_path
    recognizer_partial = instantiate(cfg.recognizer, _partial_=True)
    recognizer_ckpt_path = cfg.recognizer.ckpt_path

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
        vocab=vocab_map, use_frame_label=True, return_video_name=True, do_augm=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_collate_fn,
    )

    frame_classifier = frame_classifier_partial(
        char_size=len(char_list)+1,
    ).to(device)
    frame_classifier.load_state_dict(torch.load(frame_classifier_ckpt_path))
    frame_classifier.eval()

    recognizer = recognizer_partial(
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(recognizer_ckpt_path))
    recognizer.eval()

    frame_labels_clean = []
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            poses = batch['poses'].cuda()
            frame_idx = batch['frame_idx'].cuda()
            frame_label = batch['frame_label'].cuda()
            frame_label_pseudo = frame_label.clone()
            frame_label_pseudo[frame_label_pseudo==-1] = 0

            rlh_seg = torch.zeros_like(frame_idx)

            recognizer_out = recognizer(
                poses,
                rlh_seg,
                frame_idx,
            )
            encoder_memory = recognizer_out['encoder_memory']

            out_frame_class = frame_classifier(encoder_memory)

            valid_mask = (frame_label != -100)

            B = out_frame_class.shape[0]
            for b in range(B):
                frame_label = out_frame_class[b, valid_mask[b]].argmax(1)
                frame_labels_clean.append(frame_label.cpu().numpy())

    data = np.load(cfg.dataset.data_path, allow_pickle=True)
    data = dict(data)
    data['frame_label_clean'] = frame_labels_clean
    np.savez(cfg.dataset.data_path.replace(".npz", "_clean.npz"), **data)


if __name__ == "__main__":
    main()