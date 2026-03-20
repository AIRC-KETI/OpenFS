import os, os.path as osp
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import tqdm

import torch
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab, invert_to_chars
from lib.utils.eval import compute_acc
from lib.utils.seed import set_all_random_seed
from lib.utils.plot import save_pose_comparison_video
from lib.utils.logging import change_filehandler_mode_to_write
from lib.utils.dataset import normalize_torch
from lib.models.generator_df import EncoderTransformer
from lib.models.diffusion import Diffusion
from lib.utils.size import count_parameters

import constants

@hydra.main(version_base=None, config_path="conf", config_name="config_generator_df")
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
    diffusion_partial = instantiate(cfg.diffusion, _partial_=True)
    generator_partial = instantiate(cfg.generator, _partial_=True)
    generator_ckpt_path = cfg.generator.ckpt_path
    recognizer_partial = instantiate(cfg.recognizer, _partial_=True)
    recognizer_ckpt_path = cfg.recognizer.ckpt_path

    # == Decode == #
    chars = cfg.decode.chars

    # == Dataset == #

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
        vocab=vocab_map, use_frame_label=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_collate_fn,
    )

    diffusion: Diffusion = diffusion_partial(
        simple_loss=None,
    ).to(device)
    generator: EncoderTransformer = generator_partial(
        char_size=len(char_list)+1,
    ).to(device)
    generator.load_state_dict(torch.load(generator_ckpt_path))
    generator.eval()
    logging.info(f'Generator size: {count_parameters(generator)}')

    recognizer = recognizer_partial(
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(recognizer_ckpt_path))
    recognizer.eval()
    logging.info(f'Recognizer size: {count_parameters(recognizer)}')

    save_root = os.path.join('data', 'generated', cfg.task_type)
    os.makedirs(save_root, exist_ok=True)

    preds = []
    gt_labels = []
    with torch.no_grad():
        for trial in range(cfg.trials):
            _preds = []
            _gt_labels = []
            for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                frame_idx_gen = batch['frame_idx_gen'].cuda()
                frame_label_gen = batch['frame_label_gen'].cuda()
                word = batch['word']
                B = frame_label_gen.shape[0]

                frame_label_gen_pseudo = frame_label_gen.clone()
                frame_label_gen_pseudo[frame_label_gen_pseudo==constants.MINUS_ONE_HUNDRED_VALUE] = len(char_list)+1

                out_poses = diffusion.sampling(
                    generator,
                    frame_label_gen_pseudo,
                    frame_idx_gen,
                )
                for b in range(B):
                    out_poses[b:b+1, frame_idx_gen[b]!=1024, :, :2] \
                        = normalize_torch(out_poses[b:b+1, frame_idx_gen[b]!=1024, :, :2], normalize_value=0.5)

                for b in range(B):
                    poses_save_root = osp.join(save_root, 'poses', word[b])
                    os.makedirs(poses_save_root, exist_ok=True)
                    out_pose_np = out_poses[b, frame_idx_gen[b]!=1024].cpu().numpy()
                    np.save(osp.join(poses_save_root, f'{trial}.npy'), out_pose_np)

                rlh_seg = torch.zeros_like(frame_idx_gen)
                out_poses_2d = out_poses[..., :2]
                out_poses_2d = normalize_torch(out_poses_2d, normalize_value=0.5)
                recognizer_out = recognizer.generate(out_poses_2d, rlh_seg, frame_idx_gen, bos_token_id=len(char_list), eos_token_id=0)
                output_ids = recognizer_out['output_ids']

                for b in range(B):
                    pred = ''.join(invert_to_chars(output_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))
                    gt_label = word[b]

                    _preds.append(pred)
                    _gt_labels.append(gt_label)

            lev_acc, Ds, Ss, Is, Ns = compute_acc(_preds, _gt_labels)
            logging.info(f"Trial {trial+1} lev_acc {lev_acc:.4f}")

            preds.append(_preds)
            gt_labels.append(_gt_labels)

    preds_flatten = sum(preds, [])
    gt_labels_flatten = sum(gt_labels, [])
    lev_acc, Ds, Ss, Is, Ns = compute_acc(preds_flatten, gt_labels_flatten)
    logging.info(f"Total lev_acc {lev_acc:.4f}")


if __name__ == "__main__":
    main()