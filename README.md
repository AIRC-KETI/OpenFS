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

For FSBoard, we provide our own MediaPipe-processed results through the download scripts below.

<details>
<summary>FSBoard data issue note</summary>

The Kaggle discussion mentions potential issues in the original FSBoard release, including:

- unusual keypoint visualization where body parts appear as 3 disjoint clusters
- pose tracks with 25 landmarks instead of the expected 33
- pose visibility/presence values outside the usual `[0, 1]` range
- our interpretation: a possible schema/axis issue where `x`, `y`, and `z` may be stored in an interleaved layout rather than as standard per-joint coordinates

See the [discussion](https://www.kaggle.com/datasets/googleai/fsboard/discussion/610126) for details.
</details>
<br>

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

Preprocessing (optional):

If you want to run preprocessing yourself, place the downloaded raw datasets under `data/raw/` first:

- ChicagoFSWild download page: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm#download
- For `FSboard`, download the raw video clips with `python download_fsboard_videos.py`

- `data/raw/ChicagoFSWild`
- `data/raw/ChicagoFSWildPlus`
- `data/raw/FSboard`

For `ChicagoFSWild`:

```bash
python preprocessing.py
```

For `ChicagoFSWildPlus`:

```bash
python preprocessing.py --plus
```

For `FSboard`:

```bash
python preprocessing_fsboard.py
```

If the download scripts are temporarily blocked, you can also download the files manually from Google Drive:

- [OpenFS data folder](https://drive.google.com/drive/folders/1NB5wVgeTmoZIGjq58ueejj80mFmtebSg?usp=sharing)
- [OpenFS checkpoint folder](https://drive.google.com/drive/folders/1BiUwqlEbBhy9q4wXKEIzJ0CBuvywHGX1?usp=sharing)

Directory structure:

```text
data/
├── CustomChicagoFSWildPlusFinal.csv
├── posenet_signhand_miss.txt
├── english_word/
│   ├── Chicago_train.txt
│   ├── except_words_alpha.txt
│   ├── Neologisms.xlsx
│   ├── neobench_test.txt
│   └── words_alpha.txt
├── fingerspelling/
│   ├── Chicago_dev_merged.npz
│   ├── Chicago_dev_merged_proc.npz
│   ├── Chicago_dev_merged_proc_clean.npz
│   ├── Chicago_test_merged.npz
│   ├── Chicago_test_merged_proc.npz
│   ├── Chicago_test_merged_proc_clean.npz
│   ├── Chicago_train_merged.npz
│   ├── Chicago_train_merged_proc.npz
│   ├── Chicago_train_merged_proc_clean.npz
│   ├── ChicagoFSWild_test.npz
│   ├── ChicagoFSWildPlus_test.npz
│   ├── neobench_test_gen.npz
│   └── words_alpha_train_filtered_gen.npz
└── fingerspelling_fsboard/
    ├── fsboard_test.npz
    ├── fsboard_test_proc.npz
    ├── fsboard_test_proc_clean.npz
    ├── fsboard_train.npz
    ├── fsboard_train_proc.npz
    ├── fsboard_train_proc_clean.npz
    ├── fsboard_validation.npz
    ├── fsboard_validation_proc.npz
    └── fsboard_validation_proc_clean.npz
```

Notes:

- `CustomChicagoFSWildPlusFinal.csv` contains our manual labeling results for the Chicago test set, indicating whether each sample is fingerspelled with the left hand, right hand, or both hands.
- `proc` files contain coarse frame labels.
- `clean` files contain clean frame labels.

Checkpoint directory structure:

```text
outputs_save/
├── frame_classifier/
├── frame_classifier_fsboard/
├── generator_df/
├── generator_df_fsboard/
├── recognizer/
├── recognizer_fsboard/
└── recognizer_plus/
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
