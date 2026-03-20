python generate_data_df.py \
    generator.ckpt_path=outputs_save/generator_df/best_model.pth \
    +task_type=neobench \
    +trials=5 \
    +dataset._target_=lib.datasets.fingerspelling.WordDataset \
    +dataset.text_paths=[data/english_word/neobench_test.txt] \
    +dataset.min_frames_num=3 \
    +dataset.max_frames_num=10 \
    +dataset.min_transition_frames_num=2 \
    +dataset.max_transition_frames_num=3 \
    +dataset.use_frame_label_clean=true \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_word_dataset_fn \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    hydra.run.dir=logs/generate_neobench_data \
    --config-path=outputs_save/generator_df/.hydra \
    --config-name=config

python generate_data_df.py \
    generator.ckpt_path=outputs_save/generator_df/best_model.pth \
    +task_type=train_words_alpha_all \
    +trials=5 \
    +dataset._target_=lib.datasets.fingerspelling.WordDataset \
    +dataset.text_paths=[data/english_word/words_alpha.txt] \
    +dataset.min_frames_num=3 \
    +dataset.max_frames_num=10 \
    +dataset.min_transition_frames_num=2 \
    +dataset.max_transition_frames_num=3 \
    +dataset.use_frame_label_clean=true \
    +collate_fn._target_=lib.utils.dataloader.paired_collate_word_dataset_fn \
    recognizer.ckpt_path=outputs_save/recognizer/best_model.pth \
    hydra.run.dir=logs/generate_train_words_alpha \
    --config-path=outputs_save/generator_df/.hydra \
    --config-name=config