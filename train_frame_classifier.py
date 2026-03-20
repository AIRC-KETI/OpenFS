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

from lib.utils.text_ctc_utils import get_autoreg_vocab
from lib.utils.eval import AverageMeter
from lib.utils.time import seconds_to_hhmmss
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
    learning_rate = cfg.train.learning_rate
    optim_step_size = cfg.train.optim_step_size
    optim_gamma = cfg.train.optim_gamma
    num_epochs = cfg.train.num_epochs
    start_eval_epoch = cfg.train.start_eval_epoch
    eval_epoch_step = cfg.train.eval_epoch_step
    print_step_iter = cfg.train.print_step_iter
    batch_size = cfg.train.batch_size
    train_shuffle = cfg.train.train_shuffle
    loss_ce_fn = instantiate(cfg.train.loss.loss_ce_fn, _partial_=True)
    loss_ce_weight = cfg.train.loss.loss_ce_weight

    optimizer_partial = instantiate(cfg.train.optimizer, _partial_=True)
    train_dataset_partial = instantiate(cfg.train_dataset, _partial_=True)
    test_dataset_partial = instantiate(cfg.test_dataset, _partial_=True)

    train_dataset_collate_fn = instantiate(cfg.train_collate_fn, _partial_=True)
    test_dataset_collate_fn = instantiate(cfg.test_collate_fn, _partial_=True)

    # == Model == #
    # diffusion_partial = instantiate(cfg.diffusion, _partial_=True)
    frame_classifier_partial = instantiate(cfg.frame_classifier, _partial_=True)
    frame_classifier_ckpt_path = cfg.frame_classifier.ckpt_path
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

    frame_classifier = frame_classifier_partial(
        char_size=len(char_list)+1,
    ).to(device)
    optimizer = optimizer_partial(
        frame_classifier.parameters(), lr=learning_rate
    )

    recognizer = recognizer_partial(
        char_size=len(char_list)+1,
    ).to(device)
    recognizer.load_state_dict(torch.load(recognizer_ckpt_path))
    recognizer.eval()

    scheduler = StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

    best_loss = np.inf
    best_epoch = 0

    total_loss = AverageMeter()
    total_loss_ce = AverageMeter()

    total_test_loss = AverageMeter()
    total_test_loss_ce = AverageMeter()

    save_root = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "results")
    os.makedirs(save_root, exist_ok=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        frame_classifier.train()

        for i, batch in enumerate(train_dataloader):
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
            loss_ce = loss_ce_fn(
                out_frame_class,
                frame_label_pseudo,
                valid_mask,
                bgw=0.1
            )

            loss = loss_ce_weight*loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item(), frame_label.shape[0])
            total_loss_ce.update(loss_ce.item(), frame_label.shape[0])

            if (i+1)%print_step_iter == 0 or i+1 == len(train_dataloader):
                logging.info(
                    f'[Train] Epoch {epoch+1}/{num_epochs} Iter {i+1}/{len(train_dataloader)} ' \
                    f'- loss: {total_loss.avg:.4f} ' \
                    f'- loss_ce: {total_loss_ce.avg:.4f} ' \
                    f'- time: {seconds_to_hhmmss(time.time()-start_time)}'
                )

        total_loss.reset()
        total_loss_ce.reset()

        scheduler.step()

        if (epoch+1) < start_eval_epoch or (epoch+1) % eval_epoch_step != 0:
            continue

        frame_classifier.eval()

        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                poses = batch['poses'].cuda()
                frame_idx = batch['frame_idx'].cuda()
                frame_label = batch['frame_label'].cuda()
                frame_label_pseudo = frame_label.clone()
                frame_label_pseudo[frame_label_pseudo==-1] = 0
                word = batch['word']
                video_name = batch['video_name']

                rlh_seg = torch.zeros_like(frame_idx)

                recognizer_out = recognizer(
                    poses,
                    rlh_seg,
                    frame_idx,
                )
                encoder_memory = recognizer_out['encoder_memory']

                out_frame_class = frame_classifier(encoder_memory)

                valid_mask = (frame_label != -100)
                loss_ce = loss_ce_fn(
                    out_frame_class,
                    frame_label_pseudo,
                    valid_mask,
                    bgw=0.1
                )

                loss = loss_ce_weight*loss_ce

                total_test_loss.update(loss.item(), frame_label.shape[0])
                total_test_loss_ce.update(loss_ce.item(), frame_label.shape[0])

        cur_loss = total_test_loss.avg
        if best_loss > cur_loss:
            best_loss = cur_loss
            best_epoch = epoch+1

            torch.save(
                frame_classifier.state_dict(),
                osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, frame_classifier_ckpt_path)
            )
        logging.info(f'[Test] Epoch {epoch+1}/{num_epochs} - Cur Loss: {cur_loss:.4f} ' \
                     f'({best_epoch}, {best_loss:.4f})')

        total_test_loss.reset()
        total_test_loss_ce.reset()


if __name__ == "__main__":
    main()