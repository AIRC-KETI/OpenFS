import os, os.path as osp
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import time
import logging
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab, invert_to_chars
from lib.utils.eval import compute_acc, AverageMeter
from lib.utils.time import seconds_to_hhmmss
from lib.utils.seed import set_all_random_seed
from lib.utils.plot import save_pose_comparison_video
from lib.utils.dataset import normalize_torch
from lib.utils.logging import change_filehandler_mode_to_write
from lib.models.diffusion import Diffusion
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
    learning_rate = cfg.train.learning_rate
    optim_step_size = cfg.train.optim_step_size
    optim_gamma = cfg.train.optim_gamma
    num_epochs = cfg.train.num_epochs
    start_eval_epoch = cfg.train.start_eval_epoch
    eval_epoch_step = cfg.train.eval_epoch_step
    print_step_iter = cfg.train.print_step_iter
    batch_size = cfg.train.batch_size
    train_shuffle = cfg.train.train_shuffle
    loss_recon_joints_fn = instantiate(cfg.train.loss.loss_recon_joints_fn, _partial_=True)
    loss_recon_joints_weight = cfg.train.loss.loss_recon_joints_weight

    optimizer_partial = instantiate(cfg.train.optimizer, _partial_=True)
    train_dataset_partial = instantiate(cfg.train_dataset, _partial_=True)
    test_dataset_partial = instantiate(cfg.test_dataset, _partial_=True)

    train_dataset_collate_fn = instantiate(cfg.train_collate_fn, _partial_=True)
    test_dataset_collate_fn = instantiate(cfg.test_collate_fn, _partial_=True)

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

    dataset_train = train_dataset_partial(
        vocab=vocab_map, use_frame_label=True
    )
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=train_dataset_collate_fn,
    )

    dataset_test = test_dataset_partial(
        vocab=vocab_map, use_frame_label=True
    )
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset_collate_fn,
    )

    diffusion: Diffusion = diffusion_partial(
        simple_loss=loss_recon_joints_fn,
    ).to(device)
    generator = generator_partial(
        char_size=len(char_list)+1,
    ).to(device)
    optimizer = optimizer_partial(
        generator.parameters(), lr=learning_rate
    )

    recognizer = recognizer_partial(
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(recognizer_ckpt_path))
    recognizer.eval()

    scheduler = StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

    best_acc = 0
    best_epoch = 0

    total_loss = AverageMeter()
    total_loss_recon_joints = AverageMeter()

    save_root = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "results")
    os.makedirs(save_root, exist_ok=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        generator.train()

        for i, batch in enumerate(train_dataloader):
            frame_idx = batch['frame_idx'].cuda()
            poses = batch['poses'].cuda()

            frame_label = batch['frame_label'].cuda()
            frame_label_pseudo = frame_label.clone()
            frame_label_pseudo[frame_label_pseudo==constants.MINUS_ONE_HUNDRED_VALUE] = len(char_list)+1

            B = frame_label.shape[0]

            _, loss_recon_joints = diffusion(
                generator,
                poses,
                frame_label_pseudo,
                frame_idx,
                get_losses=True,
            )

            loss = loss_recon_joints_weight*loss_recon_joints

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item(), B)
            total_loss_recon_joints.update(loss_recon_joints.item(), B)

            if (i+1)%print_step_iter == 0 or i+1 == len(train_dataloader):
                logging.info(
                    f'[Train] Epoch {epoch+1}/{num_epochs} Iter {i+1}/{len(train_dataloader)} ' \
                    f'- loss: {total_loss.avg:.4f} ' \
                    f'- loss_recon_joints: {total_loss_recon_joints.avg:.4f} ' \
                    f'- time: {seconds_to_hhmmss(time.time()-start_time)}'
                )

        total_loss.reset()
        total_loss_recon_joints.reset()

        scheduler.step()

        if (epoch+1) < start_eval_epoch or (epoch+1) % eval_epoch_step != 0:
            continue

        generator.eval()

        preds = []
        preds_no_sampled = []
        preds_gt_poses = []
        gt_labels = []
        eval_start_time = time.time()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                frame_idx = batch['frame_idx'].cuda()
                poses = batch['poses'].cuda()
                word = batch['word']
                video_name = batch['video_name']

                frame_label = batch['frame_label'].cuda()
                frame_label_pseudo = frame_label.clone()
                frame_label_pseudo[frame_label_pseudo==constants.MINUS_ONE_HUNDRED_VALUE] = len(char_list)+1

                B = frame_label.shape[0]

                sample_poses = diffusion.sampling(
                    generator,
                    frame_label_pseudo,
                    frame_idx,
                )

                rlh_seg = torch.zeros_like(frame_idx)
                out_poses_2d = sample_poses[..., :2]
                out_poses_2d = normalize_torch(out_poses_2d, normalize_value=0.5)
                recognizer_out = recognizer.generate(out_poses_2d, rlh_seg, frame_idx, bos_token_id=len(char_list), eos_token_id=0)
                output_ids = recognizer_out['output_ids']
                for b in range(B):
                    pred = ''.join(invert_to_chars(output_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))
                    preds.append(pred)

                poses_2d = poses[..., :2]
                poses_2d = normalize_torch(poses_2d, normalize_value=0.5)
                recognizer_out = recognizer.generate(poses_2d, rlh_seg, frame_idx, bos_token_id=len(char_list), eos_token_id=0)
                output_ids_gt_poses = recognizer_out['output_ids']
                for b in range(B):
                    pred_gt_poses = ''.join(invert_to_chars(output_ids_gt_poses[b:b+1, 1:-1].cpu(), inv_vocab_map))
                    preds_gt_poses.append(pred_gt_poses)
                    gt_label = word[b]
                    gt_labels.append(gt_label)

                if i < 2:
                    save_pose_comparison_video(
                        poses_2d[0].cpu(), out_poses_2d[0].cpu(), word[0],
                        os.path.join(
                            save_root, f"epoch_{epoch+1}",
                            f"gen_{i}_{int(os.environ.get('LOCAL_RANK', 0))}.mp4"
                        )
                    )

        lev_acc, Ds, Ss, Is, Ns = compute_acc(preds, gt_labels)
        lev_acc_gt_poses, Ds_gt_poses, Ss_gt_poses, Is_gt_poses, Ns_gt_poses = compute_acc(preds_gt_poses, gt_labels)

        if best_acc < lev_acc:
            best_acc = lev_acc
            best_epoch = epoch+1

            torch.save(
                generator.state_dict(),
                osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, generator_ckpt_path)
            )
        logging.info(f'[Test] Epoch {epoch+1}/{num_epochs} - Cur Acc: {lev_acc:.4f} Cur GT Acc: {lev_acc_gt_poses:.4f} ({best_epoch}, {best_acc:.4f})')


if __name__ == "__main__":
    main()