import os, os.path as osp
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import time
import logging
import tqdm

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.text_ctc_utils import get_autoreg_vocab, invert_to_chars
from lib.utils.eval import compute_acc, AverageMeter
from lib.utils.time import seconds_to_hhmmss
from lib.utils.seed import set_all_random_seed
from lib.utils.logging import change_filehandler_mode_to_write
import constants

@hydra.main(version_base=None, config_path="conf", config_name="config_recognizer")
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

    # Loss
    loss_dec_fn = instantiate(cfg.train.loss.loss_dec_fn, _partial_=True)
    loss_dec_weight = cfg.train.loss.loss_dec_weight
    loss_attn_ent_fn = instantiate(cfg.train.loss.loss_attn_ent_fn, _partial_=True)
    loss_attn_ent_weight = cfg.train.loss.loss_attn_ent_weight
    loss_attn_ent_warm_epoch = cfg.train.loss.loss_attn_ent_warm_epoch
    loss_monotonicity_fn = instantiate(cfg.train.loss.loss_monotonicity_fn, _partial_=True)
    loss_monotonicity_weight = cfg.train.loss.loss_monotonicity_weight
    loss_monotonicity_warm_epoch = cfg.train.loss.loss_monotonicity_warm_epoch

    optimizer_partial = instantiate(cfg.train.optimizer, _partial_=True)

    # == Dataset == #
    train_dataset_partial = instantiate(cfg.train_dataset, _partial_=True)
    test_dataset_partial = instantiate(cfg.test_dataset, _partial_=True)

    train_dataset_collate_fn = instantiate(cfg.train_collate_fn, _partial_=True)
    test_dataset_collate_fn = instantiate(cfg.test_collate_fn, _partial_=True)

    # == Model == #
    recognizer_partial = instantiate(cfg.recognizer, _partial_=True)
    recognizer_ckpt_path = cfg.recognizer.ckpt_path

    # == Decode == #
    chars = cfg.decode.chars

    # == use_generator == #
    iterations = cfg.train.iterations
    gen_batch_size = cfg.train.gen_batch_size
    gen_train_dataset_partial = instantiate(cfg.gen_train_dataset, _partial_=True)
    gen_train_dataset_collate_fn = instantiate(cfg.gen_train_collate_fn, _partial_=True)

    # ===== End Configurations ===== #

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f'device {device}')

    vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)

    logging.info(vocab_map)
    logging.info(inv_vocab_map)
    logging.info(char_list)

    if batch_size > 0:
        dataset_train = train_dataset_partial(
            vocab=vocab_map,
        )
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=train_dataset_collate_fn,
        )
        train_iter = iter(train_dataloader)
    else:
        train_dataloader = None
        train_iter = None

    if gen_batch_size > 0:
        gen_dataset_train = gen_train_dataset_partial(
            vocab=vocab_map,
        )
        gen_train_dataloader = DataLoader(
            gen_dataset_train,
            batch_size=gen_batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=gen_train_dataset_collate_fn,
        )
        gen_train_iter = iter(gen_train_dataloader)
    else:
        gen_train_dataloader = None
        gen_train_iter = None

    dataset_test = test_dataset_partial(vocab=vocab_map)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset_collate_fn,
    )

    recognizer = recognizer_partial(
        encoder_out_dim=len(char_list),
        char_size=len(char_list)+1,
    ).to(device)
    optimizer = optimizer_partial(
        recognizer.parameters(), lr=learning_rate
    )

    if 'resume_ckpt_path' in cfg.recognizer:
        recognizer.load_state_dict(torch.load(cfg.recognizer.resume_ckpt_path))

    scheduler = StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

    best_acc = 0
    best_epoch = 0

    total_loss = AverageMeter()
    total_loss_dec = AverageMeter()
    total_loss_attn_ent = AverageMeter()
    total_loss_monotonicity = AverageMeter()

    start_time = time.time()
    for epoch in range(num_epochs):
        recognizer.train()

        for i in range(iterations):
            if batch_size > 0:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                poses = batch['poses'].cuda()
                rlh_seg = batch['rlh_seg'].cuda()
                frame_idx = batch['frame_idx'].cuda()
                target_ids = batch['target_ids'].cuda()

                B, L_poses, J, _ = poses.shape
                _, L_tg = target_ids.shape[:2]
            else:
                poses = rlh_seg = frame_idx = target_ids = None
                B = L_poses = L_tg = 0

            if gen_batch_size > 0:
                try:
                    gen_batch = next(gen_train_iter)
                except StopIteration:
                    gen_train_iter = iter(gen_train_dataloader)
                    gen_batch = next(gen_train_iter)

                poses_gen = gen_batch['poses'].cuda()
                rlh_seg_gen = gen_batch['rlh_seg'].cuda()
                frame_idx_gen = gen_batch['frame_idx'].cuda()
                target_ids_gen = gen_batch['target_ids'].cuda()

                B_ft, L_poses_gen, J, _ = poses_gen.shape
                L_tg_gen = target_ids_gen.shape[1]
            else:
                poses_gen = rlh_seg_gen = frame_idx_gen = target_ids_gen = None
                B_ft = L_poses_gen = L_tg_gen = 0

            L_max = max(L_poses, L_poses_gen)
            joint_dim = recognizer.input_dim // J if B + B_ft > 0 else 2  # fallback value

            poses_combined = torch.ones(B+B_ft, L_max, 21, joint_dim).cuda() * constants.MINUS_TWO_VALUE
            rlh_seg_combined = torch.ones(B+B_ft, L_max).long().cuda() * 32
            frame_idx_combined = torch.ones(B+B_ft, L_max).long().cuda() * 1024

            if B > 0:
                poses_combined[:B, :L_poses] = poses[..., :joint_dim]
                rlh_seg_combined[:B, :L_poses] = rlh_seg
                frame_idx_combined[:B, :L_poses] = frame_idx

            if B_ft > 0:
                poses_combined[B:, :L_poses_gen] = poses_gen[..., :joint_dim]
                rlh_seg_combined[B:, :L_poses_gen] = rlh_seg_gen
                frame_idx_combined[B:, :L_poses_gen] = frame_idx_gen

            poses = poses_combined
            rlh_seg = rlh_seg_combined
            frame_idx = frame_idx_combined

            L_max_tg = max(L_tg, L_tg_gen)
            target_ids_combined = torch.ones(B+B_ft, L_max_tg).long().cuda() * constants.MINUS_ONE_HUNDRED_VALUE
            if B > 0:
                target_ids_combined[:B, :L_tg] = target_ids
            if B_ft > 0:
                target_ids_combined[B:, :L_tg_gen] = target_ids_gen

            target_ids = target_ids_combined

            target_ids_pseudo = target_ids.clone()
            target_ids_pseudo[target_ids_pseudo==constants.MINUS_ONE_HUNDRED_VALUE] = len(char_list)+1

            recognizer_out = recognizer(
                poses,
                rlh_seg,
                frame_idx,
                target_ids_pseudo[:, :-1]
            )
            logits = recognizer_out['logits']
            B, L_poses = poses.shape[:2]
            _, L_tg = target_ids.shape[:2]

            if loss_dec_weight > 0:
                valid_mask = (target_ids[:, 1:] != -100) # Not poses
                loss_dec = loss_dec_fn(
                    logits.transpose(-1, -2),
                    target_ids[:, 1:],
                    valid_mask
                )
            else:
                loss_dec = torch.zeros([1], device=device)

            if (epoch+1) > loss_attn_ent_warm_epoch and loss_attn_ent_weight > 0:
                cross_attns = torch.cat(recognizer_out['cross_attns'], dim=1).reshape(B, cfg.recognizer.num_layers, L_tg-1, L_poses) # 3
                loss_attn_ent = loss_attn_ent_fn(cross_attns, rlh_seg, target_ids)
            else:
                loss_attn_ent = torch.zeros([1], device=device)

            if (epoch+1) > loss_monotonicity_warm_epoch and loss_monotonicity_weight > 0.0:
                cross_attns = torch.cat(recognizer_out['cross_attns'], dim=1).reshape(B, cfg.recognizer.num_layers, L_tg-1, L_poses) # 3
                loss_monotonicity = loss_monotonicity_fn(cross_attns, rlh_seg, target_ids)
            else:
                loss_monotonicity = torch.zeros([1], device=device)

            loss = loss_dec_weight*loss_dec \
                 + loss_attn_ent_weight*loss_attn_ent \
                 + loss_monotonicity_weight*loss_monotonicity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item(), target_ids.shape[0])
            total_loss_dec.update(loss_dec.item(), target_ids.shape[0])
            total_loss_attn_ent.update(loss_attn_ent.item(), target_ids.shape[0])
            total_loss_monotonicity.update(loss_monotonicity.item(), target_ids.shape[0])

            if (i+1)%print_step_iter == 0 or i+1 == iterations:
                logging.info(
                    f'Epoch {epoch+1}/{num_epochs} Iter {i+1}/{iterations} ' \
                    f'- loss: {total_loss.avg:.4f} ' \
                    f'- loss_dec: {total_loss_dec.avg:.4f} ' \
                    f'- loss_attn_ent: {total_loss_attn_ent.avg:.4f} ' \
                    f'- loss_monotonicity: {total_loss_monotonicity.avg:.4f} ' \
                    f'- time: {seconds_to_hhmmss(time.time()-start_time)}'
                )

        total_loss.reset()
        total_loss_dec.reset()
        total_loss_attn_ent.reset()
        total_loss_monotonicity.reset()

        scheduler.step()

        if (epoch+1) < start_eval_epoch or (epoch+1) % eval_epoch_step != 0:
            continue

        recognizer.eval()

        preds = []
        gt_labels = []

        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                rlh_seg = batch['rlh_seg'].cuda()
                poses = batch['poses'].cuda()
                target_ids = batch['target_ids'].cuda()
                frame_idx = batch['frame_idx'].cuda()
                video_name = batch['video_name']

                recognizer_out = recognizer.generate(poses, rlh_seg, frame_idx, bos_token_id=len(char_list), eos_token_id=0)
                output_ids = recognizer_out['output_ids']

                B = poses.shape[0]
                for b in range(B):
                    current_pred = ''.join(invert_to_chars(output_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))
                    gt_label = ''.join(invert_to_chars(target_ids[b:b+1, 1:-1].cpu(), inv_vocab_map))

                    preds.append(current_pred)
                    gt_labels.append(gt_label)

                # logging.info(f'{current_pred} \t {current_pred_ctc} \t {gt_label} {video_name}  {i+1}/{len(test_dataloader)}  {seconds_to_hhmmss(time.time()-eval_start_time)}')

        lev_acc, Ds, Ss, Is, Ns = compute_acc(preds, gt_labels)

        if best_acc < lev_acc:
            best_acc = lev_acc
            best_epoch = epoch+1

            torch.save(
                recognizer.state_dict(),
                osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, recognizer_ckpt_path.replace('.pth', f'{best_epoch}.pth'))
            )
            torch.save(
                recognizer.state_dict(),
                osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, recognizer_ckpt_path)
            )
        logging.info(f'[Test] Epoch {epoch+1}/{num_epochs} - Letter Acc: {lev_acc:.4f} - Deletion: {Ds} - Substitution: {Ss} - Insertion: {Is} - Best Acc {best_acc:.4f} ({best_epoch})')


if __name__ == "__main__":
    main()