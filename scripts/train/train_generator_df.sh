#!/bin/bash

MODEL=${1:-all}

train_generator_df () {
echo outputs_save/generator_df
python train_generator_df.py \
    train.num_workers=16 \
    train.batch_size=20 \
    train.num_epochs=1000 \
    train.optim_step_size=10000 \
    train.optim_gamma=0.1 \
    train.learning_rate=1e-4 \
    train.start_eval_epoch=100 \
    train.eval_epoch_step=100 \
    train_dataset.data_path=data/fingerspelling/Chicago_train_merged_proc_clean.npz \
    train_dataset.representation=3d \
    train_dataset.do_affine=false \
    train_dataset.do_reverse=false \
    test_dataset.data_path=data/fingerspelling/Chicago_dev_merged_proc_clean.npz \
    test_dataset.representation=3d \
    generator.pose_dim=63 \
    generator.dim_feedforward=1024 \
    generator.num_layers=8 \
    generator.nhead=4 \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    hydra.run.dir=outputs_save/generator_df
}

train_generator_df_fsboard () {
echo outputs_save/generator_df_fsboard
python train_generator_df.py \
    train.num_workers=16 \
    train.batch_size=20 \
    train.num_epochs=1000 \
    train.optim_step_size=10000 \
    train.optim_gamma=0.1 \
    train.learning_rate=1e-4 \
    train.start_eval_epoch=100 \
    train.eval_epoch_step=100 \
    train_dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    train_dataset.data_path=data/fingerspelling/fsboard_train_proc_clean.npz \
    train_dataset.representation=3d \
    train_dataset.do_affine=false \
    train_dataset.do_reverse=false \
    test_dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    test_dataset.data_path=data/fingerspelling/fsboard_validation_proc_clean.npz \
    test_dataset.representation=3d \
    generator.pose_dim=63 \
    generator.dim_feedforward=1024 \
    generator.num_layers=8 \
    generator.nhead=4 \
    decode=fsboard \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    hydra.run.dir=outputs_save/generator_df_fsboard
}

if [[ $MODEL == "generator_df" || $MODEL == "all" ]]; then
    train_generator_df
fi

if [[ $MODEL == "generator_df_fsboard" || $MODEL == "all" ]]; then
    train_generator_df_fsboard
fi