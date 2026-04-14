import os
import json
import shutil
import kagglehub

base = "googleai/fsboard"
BASE_DIR = "data/raw/FSboard"

datasets = ["daun_v3", "dmk_v3"]

for dataset in datasets:

    splits = ["train", "validation", "test"] if dataset == "daun_v3" else ["train"]

    for split in splits:

        metadata_path = os.path.join(BASE_DIR, dataset, "metadata", f"{dataset}-{split}.json")
        if not os.path.exists(metadata_path):
            continue

        kaggle_split = f"{dataset}-{split}"
        save_root = os.path.join(BASE_DIR, dataset, "video_clips", kaggle_split)
        os.makedirs(save_root, exist_ok=True)

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        for item in meta:

            clip_filename = item["clipFilename"]
            kaggle_path = f"{dataset}/video_clips/{kaggle_split}/{clip_filename}"
            dst_path = os.path.join(save_root, clip_filename)

            if os.path.exists(dst_path):
                print(f"[SKIP] {dst_path}")
                continue

            try:
                downloaded_path = kagglehub.dataset_download(
                    base,
                    path=kaggle_path,
                    force_download=False
                )

                if downloaded_path and os.path.exists(downloaded_path):
                    shutil.move(downloaded_path, dst_path)
                    print(f"Moved to: {dst_path}")
                else:
                    print(f"[PASS] Download failed: {kaggle_path}")

            except Exception as e:
                print(f"[PASS] Error: {kaggle_path} | {e}")
                continue