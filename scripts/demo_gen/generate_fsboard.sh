python demo_generation.py \
    generator.ckpt_path=outputs_save/generator_df_fsboard/best_model.pth \
    generator_global.ckpt_path=null \
    generator_word.ckpt_path=null \
    recognizer.ckpt_path=outputs_save/recognizer_fsboard/best_model.pth \
    decode=fsboard \
    hydra.run.dir=./demo/generate_fsboard \
    hydra.output_subdir=null