import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

import torch

from lib.utils.text_ctc_utils import (
    get_autoreg_vocab,
    invert_to_chars,
)
from lib.utils.eval import compute_acc
from lib.utils.seed import set_all_random_seed
from lib.utils.plot import render_sequence_2d
from lib.utils.logging import change_filehandler_mode_to_write
from lib.utils.dataset import (
    normalize_torch,
    preprocess_word,
    make_frame_labels_wo_zero
)
from lib.models.generator_df import EncoderTransformer
from lib.models.diffusion import Diffusion
import constants

def normalize_bone_length(
    poses: torch.Tensor,
    target_length_0_9: float = 0.5,
    target_length_5_17: float = 0.5,
    root_center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize so that:
      - (0–9) 3D bone length ≈ target_length_0_9  (per frame)
      - (5–17) x-axis bone length ≈ target_length_5_17_x  (per frame)

    Args:
        poses (torch.Tensor): (B, T, J, 3)
    """
    assert poses.dim() == 4 and poses.size(-1) == 3
    B, T, J, _ = poses.shape
    assert J > 17

    out = poses.clone()

    if root_center:
        out = out - out[..., 0:1, :]  # move joint 0 to origin

    # (1) normalize 0–9 3D bone length per frame
    diff_0_9 = out[..., 0, :] - out[..., 9, :]  # (B, T, 3)
    L_0_9 = diff_0_9.norm(dim=-1)               # (B, T)
    s_0_9 = target_length_0_9 / (L_0_9 + eps)   # (B, T)
    s_0_9 = s_0_9.unsqueeze(-1).unsqueeze(-1)   # (B, T, 1, 1)
    out = out * s_0_9

    # (2) normalize 5–17 3D bone length per frame
    diff_5_17 = out[..., 5, :] - out[..., 17, :]
    L_5_17 = diff_5_17.norm(dim=-1)
    s2 = target_length_5_17 / (L_5_17 + eps)
    s2 = s2.unsqueeze(-1).unsqueeze(-1)
    out = out * s2

    return out


@hydra.main(version_base=None, config_path="conf", config_name="config_demo_generator_df")
def main(cfg: DictConfig):
    root_logger = logging.getLogger()
    change_filehandler_mode_to_write(root_logger)

    logging.info(OmegaConf.to_yaml(cfg))

    set_all_random_seed(cfg.seed)

    # ===== Begin Configurations ===== #
    # == Model == #
    diffusion_partial = instantiate(cfg.diffusion, _partial_=True)
    diffusion_global_partial = instantiate(cfg.diffusion_global, _partial_=True)
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

    diffusion: Diffusion = diffusion_partial(
        simple_loss=None,
    ).to(device)
    diffusion_global: Diffusion = diffusion_global_partial(
        simple_loss=None,
    ).to(device)
    generator: EncoderTransformer = generator_partial(
        char_size=len(char_list)+1,
    ).to(device)
    generator.load_state_dict(torch.load(generator_ckpt_path))
    generator.eval()

    recognizer = recognizer_partial(
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(recognizer_ckpt_path))
    recognizer.eval()

    save_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    with torch.no_grad():
        word = input('Word:')
        set_all_random_seed(cfg.seed)

        _, ext_syllable_indices = preprocess_word(word, vocab_map)
        num_syllables = len(ext_syllable_indices[1:-1])

        num_frames = []
        for _ in range(num_syllables):
            num_frames.append(8)
        frame_label_gen = make_frame_labels_wo_zero(ext_syllable_indices, num_frames, vocab_map)

        word = [word]
        frame_idx_gen = torch.arange(len(frame_label_gen)).unsqueeze(0).cuda()
        frame_label_gen = torch.LongTensor(frame_label_gen).unsqueeze(0).cuda()
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

        out_poses_norm = normalize_bone_length(out_poses, 0.5, 0.4, True)
        for b in range(B):
            labels = ''.join(invert_to_chars(frame_label_gen[b:b+1].cpu(), inv_vocab_map))
            video_save_root = osp.join(save_root, 'video', word[b])
            render_sequence_2d(out_poses_norm[b].cpu().numpy(), osp.join(video_save_root, str(b)), labels)

        rlh_seg = torch.zeros_like(frame_idx_gen)
        out_poses_2d = out_poses[..., :2]
        recognizer_out = recognizer.generate(out_poses_2d, rlh_seg, frame_idx_gen, bos_token_id=len(char_list), eos_token_id=0)
        output_ids_local = recognizer_out['output_ids']
        print(invert_to_chars(output_ids_local[0:0+1, 1:-1].cpu(), inv_vocab_map))
        print(compute_acc([invert_to_chars(output_ids_local[0:0+1, 1:-1].cpu(), inv_vocab_map)], ['denver']))

if __name__ == "__main__":
    main()