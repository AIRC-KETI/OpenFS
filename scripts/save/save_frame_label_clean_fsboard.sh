python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier_fsboard/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_train_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier_fsboard/.hydra \
    --config-name=config

python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier_fsboard/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_validation_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier_fsboard/.hydra \
    --config-name=config

python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier_fsboard/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/fsboard_test_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier_fsboard/.hydra \
    --config-name=config