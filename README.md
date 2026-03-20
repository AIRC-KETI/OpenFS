# OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis

🎉 Accepted to CVPR 2026.

[ArXiv](https://arxiv.org/pdf/2602.22949) | [Project Page](https://junukcha.github.io/project/openfs/)

## License

This project is released for **non-commercial research purposes only**.

For any commercial use, please contact the authors to obtain permission.

First author: junukcha@gmail.com ([Junuk Cha](https://junukcha.github.io/))

Co-author: jihyeonk0226@gmail.com ([Jihyeon Kim](https://sites.google.com/view/jihyeonkim2/%ED%99%88))

Corresponding author: hanmu@keti.re.kr or hanmupark@gmail.com ([Han-Mu Park](https://sites.google.com/view/hanmupark))

## Installation

Tested environment: Ubuntu 20.04, CUDA 12.4, NVIDIA A40.

```bash
source scripts/install.sh
```

## Data

Training data (optional, only needed for training):

```bash
source scripts/download/download_data_train.sh
```

Note: this step also downloads generated training data and may take a while.

Test data:

```bash
source scripts/download/download_data_test.sh
```

Pretrained checkpoints:

```bash
source scripts/download/download_ckpts.sh
```

Example 1. Evaluation only:

```bash
source scripts/download/download_data_test.sh
source scripts/download/download_ckpts.sh
```

Example 2. Training + evaluation:

```bash
source scripts/download/download_data_train.sh
source scripts/download/download_data_test.sh
source scripts/download/download_ckpts.sh
```

## Test

### Chicago

Recognizer:

```bash
source scripts/test/test_recognizer.sh recognizer
source scripts/test/test_recognizer.sh recognizer_plus
```

Hand detection:

```bash
source scripts/test/test_hand_detect.sh recognizer
source scripts/test/test_hand_detect.sh recognizer_plus
```

### FSBoard

Recognizer:

```bash
source scripts/test/test_recognizer.sh recognizer_fsboard
```

## Demo (Generation)

### Chicago

Run the demo generator:

```bash
source scripts/demo_gen/generate.sh
```

### FSBoard

Run the demo generator:

```bash
source scripts/demo_gen/generate_fsboard.sh
```

### `img2vid`

Convert generated image frames to a video:

```bash
source scripts/demo_gen/img2vid.sh "<input_pattern>" <output_video> [fps]
```

Example:

```bash
source scripts/demo_gen/img2vid.sh "demo/generate/video/denver/0/index_%04d.png" demo/generate/video/denver/0.mp4 15
```

## Training

### Chicago

Train the recognizer:

```bash
source scripts/train/train_recognizer.sh recognizer
```

Save weak frame labels:

```bash
source scripts/save/save_hand_detect_weak_frame_label.sh
```

Train the frame classifier:

```bash
source scripts/train/train_frame_classifier.sh frame_classifier
```

Save clean frame labels:

```bash
source scripts/save/save_frame_label_clean.sh
```

Train the generator:

```bash
source scripts/train/train_generator_df.sh generator_df
```

Generate synthetic data:

```bash
source scripts/generate/generate_data_df.sh
python make_npz_gen.py
```

Train the recognizer with the generated data:

```bash
source scripts/train/train_recognizer.sh recognizer_plus
```

### FSBoard

Train the recognizer:

```bash
source scripts/train/train_recognizer.sh recognizer_fsboard
```

Save weak frame labels:

```bash
source scripts/save/save_hand_detect_weak_frame_label_fsboard.sh
```

Train the frame classifier:

```bash
source scripts/train/train_frame_classifier.sh frame_classifier_fsboard
```

Save clean frame labels:

```bash
source scripts/save/save_frame_label_clean_fsboard.sh
```

Train the generator:

```bash
source scripts/train/train_generator_df.sh generator_df_fsboard
```