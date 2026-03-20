#!/bin/bash

MODEL=${1:-all}


run_recognizer () {
MODEL_PATH=$1
NAME=$2

# fswild
python test_recognizer.py \
    recognizer.ckpt_path=$MODEL_PATH/best_model.pth \
    +test_batch_size=32 \
    test_dataset.data_path=data/fingerspelling/ChicagoFSWild_test.npz \
    hydra.run.dir=test_results/base/fswild/$NAME \
    --config-path=$MODEL_PATH/.hydra \
    --config-name=config

# fswildplus
python test_recognizer.py \
    recognizer.ckpt_path=$MODEL_PATH/best_model.pth \
    +test_batch_size=32 \
    test_dataset.data_path=data/fingerspelling/ChicagoFSWildPlus_test.npz \
    hydra.run.dir=test_results/base/fswildplus/$NAME \
    --config-path=$MODEL_PATH/.hydra \
    --config-name=config

# neobench
python test_recognizer.py \
    recognizer.ckpt_path=$MODEL_PATH/best_model.pth \
    test_dataset._target_=lib.datasets.fingerspelling.DGDataset \
    +test_dataset.data_paths=[data/fingerspelling/neobench_test_gen.npz] \
    test_dataset.return_video_name=false \
    hydra.run.dir=test_results/base/neobench/$NAME \
    --config-path=$MODEL_PATH/.hydra \
    --config-name=config
}


run_recognizer_fsboard () {
MODEL_PATH=$1
NAME=$2

python test_recognizer_fsboard.py \
    recognizer.ckpt_path=$MODEL_PATH/best_model.pth \
    +test_batch_size=32 \
    test_dataset.data_path=data/fingerspelling_fsboard/fsboard_test.npz \
    hydra.run.dir=test_results/base/fsboard/$NAME \
    --config-path=$MODEL_PATH/.hydra \
    --config-name=config
}


if [[ $MODEL == "recognizer" || $MODEL == "all" ]]; then
    run_recognizer outputs_save/recognizer recognizer
fi

if [[ $MODEL == "recognizer_plus" || $MODEL == "all" ]]; then
    run_recognizer outputs_save/recognizer_plus recognizer_plus
fi

if [[ $MODEL == "recognizer_fsboard" || $MODEL == "all" ]]; then
    run_recognizer_fsboard outputs_save/recognizer_fsboard recognizer_fsboard
fi