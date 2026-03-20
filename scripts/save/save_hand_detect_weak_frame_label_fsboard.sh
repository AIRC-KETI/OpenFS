python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_train.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer_fsboard/.hydra \
    --config-name=config

python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_validation.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer_fsboard/.hydra \
    --config-name=config

python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_test.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer_fsboard/.hydra \
    --config-name=config