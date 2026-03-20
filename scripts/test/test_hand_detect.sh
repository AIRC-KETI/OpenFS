#!/bin/bash

MODEL=${1:-all}

run_hand_detect () {
python test_hand_detect.py \
    recognizer.ckpt_path=$1/best_model.pth \
    test_dataset.data_path=data/fingerspelling/ChicagoFSWild_test.npz \
    hydra.run.dir=test_results/hand_detect/$2 \
    --config-path=$1/.hydra \
    --config-name=config
}

if [[ $MODEL == "recognizer" || $MODEL == "all" ]]; then
    run_hand_detect outputs_save/recognizer recognizer
fi

if [[ $MODEL == "recognizer_plus" || $MODEL == "all" ]]; then
    run_hand_detect outputs_save/recognizer_plus recognizer_plus
fi