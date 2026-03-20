#!/bin/bash

MODEL=${1:-all}


train_recognizer () {
echo outputs_save/recognizer
python train_recognizer.py \
    train.optim_step_size=10 \
    train.optim_gamma=0.1 \
    train.num_epochs=20 \
    train.batch_size=64 \
    train.loss.loss_attn_ent_weight=0.8 \
    train.loss.loss_monotonicity_weight=1.0 \
    train_dataset.representation=2d \
    test_dataset.representation=2d \
    recognizer.input_dim=42 \
    hydra.run.dir="outputs_save/recognizer"
}


train_recognizer_plus () {
echo outputs_save/recognizer_plus
python train_recognizer_w_gen.py \
    +train.iterations=3600 \
    train.learning_rate=1e-4 \
    train.optim_step_size=20 \
    train.optim_gamma=0.1 \
    train.num_epochs=30 \
    train.start_eval_epoch=30 \
    train.eval_epoch_step=1 \
    train.batch_size=32 \
    train.gen_batch_size=32 \
    train.loss.loss_attn_ent_weight=0.8 \
    train.loss.loss_attn_ent_warm_epoch=5 \
    train.loss.loss_monotonicity_weight=1.0 \
    train.loss.loss_monotonicity_warm_epoch=5 \
    train_dataset.representation=2d \
    gen_train_dataset.data_paths=[data/fingerspelling/words_alpha_train_filtered_gen.npz] \
    gen_train_dataset.representation=2d \
    test_dataset.representation=2d \
    recognizer.input_dim=42 \
    hydra.run.dir="outputs_save/recognizer_plus"
}


train_recognizer_fsboard () {
echo outputs_save/recognizer_fsboard
python train_recognizer.py \
    train.optim_step_size=80 \
    train.optim_gamma=0.1 \
    train.num_epochs=100 \
    train.batch_size=64 \
    train.start_eval_epoch=80 \
    train.loss.loss_attn_ent_weight=0.8 \
    train.loss.loss_monotonicity_weight=1.0 \
    train_dataset.data_path=data/fingerspelling_fsboard/fsboard_train.npz \
    train_dataset.representation=2d \
    test_dataset.data_path=data/fingerspelling_fsboard/fsboard_validation.npz \
    test_dataset.representation=2d \
    recognizer.input_dim=42 \
    decode=fsboard \
    hydra.run.dir="outputs_save/recognizer_fsboard"
}


if [[ $MODEL == "recognizer" || $MODEL == "all" ]]; then
    train_recognizer
fi

if [[ $MODEL == "recognizer_plus" || $MODEL == "all" ]]; then
    train_recognizer_plus
fi

if [[ $MODEL == "recognizer_fsboard" || $MODEL == "all" ]]; then
    train_recognizer_fsboard
fi