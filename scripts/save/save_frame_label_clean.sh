python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_train_merged_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier/.hydra \
    --config-name=config

python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_dev_merged_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier/.hydra \
    --config-name=config

python save_frame_label_clean.py \
    frame_classifier.ckpt_path=outputs_save/frame_classifier/best_model.pth \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    +dataset._target_=lib.datasets.fingerspelling_gen.MediaPipeDataset \
    +dataset.data_path=data/fingerspelling/Chicago_test_merged_proc.npz \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_gen_fn \
    --config-path=outputs_save/frame_classifier/.hydra \
    --config-name=config
