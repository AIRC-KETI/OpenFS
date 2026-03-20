#!/bin/bash

MODEL=${1:-all}

train_frame_classifier () {
echo outputs_save/frame_classifier
python train_frame_classifier.py \
    train.optim_step_size=10 \
    train.optim_gamma=0.1 \
    train.num_epochs=20 \
    train.start_eval_epoch=10 \
    train.eval_epoch_step=1 \
    train_dataset.representation=2d \
    train_dataset.data_path=data/fingerspelling/Chicago_train_merged_proc.npz \
    test_dataset.representation=2d \
    test_dataset.data_path=data/fingerspelling/Chicago_dev_merged_proc.npz \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    hydra.run.dir=outputs_save/frame_classifier
}

train_frame_classifier_fsboard () {
echo outputs_save/frame_classifier_fsboard
python train_frame_classifier.py \
    train.optim_step_size=10 \
    train.optim_gamma=0.1 \
    train.num_epochs=20 \
    train.start_eval_epoch=10 \
    train.eval_epoch_step=1 \
    train_dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    train_dataset.representation=2d \
    train_dataset.data_path=data/fingerspelling/fsboard_train_proc.npz \
    test_dataset._target_=lib.datasets.fingerspelling_gen_fsboard.MediaPipeDataset \
    test_dataset.representation=2d \
    test_dataset.data_path=data/fingerspelling/fsboard_test_proc.npz \
    decode=fsboard \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    hydra.run.dir=outputs_save/frame_classifier_fsboard
}

if [[ $MODEL == "frame_classifier" || $MODEL == "all" ]]; then
    train_frame_classifier
fi

if [[ $MODEL == "frame_classifier_fsboard" || $MODEL == "all" ]]; then
    train_frame_classifier_fsboard
fi