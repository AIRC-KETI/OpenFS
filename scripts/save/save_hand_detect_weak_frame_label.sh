python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_train_merged.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer/.hydra \
    --config-name=config

python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_dev_merged.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer/.hydra \
    --config-name=config

python save_hand_detect_weak_frame_label.py \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_test_merged.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_recog_fn \
    --config-path=outputs_save/recognizer/.hydra \
    --config-name=config