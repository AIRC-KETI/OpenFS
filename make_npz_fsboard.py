import os, os.path as osp
import numpy as np
import json
import tqdm


def process(dataset, split):
    metadata_path = osp.join(dataset, "metadata", f"{dataset}-{split}.json")
    base_folder = osp.join(dataset, "mediapipe_results", split)

    if not osp.exists(metadata_path):
        return [], {}

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    samples = []

    stats = {
        "total_meta": 0,
        "skip_no_folder": 0,
        "skip_no_word": 0,
        "skip_no_hands": 0,
        "kept": 0,
    }

    for item in tqdm.tqdm(meta, desc=f"{dataset}-{split}"):

        stats["total_meta"] += 1

        clip_name = osp.splitext(item["clipFilename"])[0]
        save_folder = osp.join(base_folder, clip_name)

        if not osp.exists(save_folder):
            stats["skip_no_folder"] += 1
            continue

        right_path = osp.join(save_folder, "right_hand_poses.npy")
        left_path = osp.join(save_folder, "left_hand_poses.npy")
        right_idx_path = osp.join(save_folder, "right_frame_indices.npy")
        left_idx_path = osp.join(save_folder, "left_frame_indices.npy")
        word_path = osp.join(save_folder, "word.npy")

        if not osp.exists(word_path):
            stats["skip_no_word"] += 1
            continue

        right_arr = np.load(right_path) if osp.exists(right_path) else np.zeros((0, 21, 3), dtype=np.float32)
        left_arr = np.load(left_path) if osp.exists(left_path) else np.zeros((0, 21, 3), dtype=np.float32)
        right_idx = np.load(right_idx_path) if osp.exists(right_idx_path) else np.zeros((0,), dtype=np.int16)
        left_idx = np.load(left_idx_path) if osp.exists(left_idx_path) else np.zeros((0,), dtype=np.int16)
        word = np.load(word_path)

        if len(right_arr) == 0 and len(left_arr) == 0:
            stats["skip_no_hands"] += 1
            continue

        samples.append(
            (
                clip_name,
                right_arr,
                left_arr,
                right_idx,
                left_idx,
                str(word),
                split,
            )
        )

        stats["kept"] += 1

    return samples, stats


def main():

    all_samples = []
    all_stats = {}

    def run(ds, sp):
        s, st = process(ds, sp)
        all_samples.extend(s)
        all_stats[f"{ds}-{sp}"] = st

    for split in ["train", "validation", "test"]:
        run("daun_v3", split)

    run("dmk_v3", "train")

    train = [x for x in all_samples if x[6] == "train"]
    val = [x for x in all_samples if x[6] == "validation"]
    test = [x for x in all_samples if x[6] == "test"]

    def save(split_name, data):

        if len(data) == 0:
            return

        vidnames = []
        right_list = []
        left_list = []
        right_idx_list = []
        left_idx_list = []
        words = []

        for clip, r, l, ri, li, w, _ in data:
            vidnames.append(clip)
            right_list.append({0: r})
            left_list.append({0: l})
            right_idx_list.append(ri)
            left_idx_list.append(li)
            words.append(w)

        save_path = f"fsboard_{split_name}.npz"

        np.savez(
            save_path,
            vidnames=np.array(vidnames, dtype=object),
            right_hand_poses=np.array(right_list, dtype=object),
            left_hand_poses=np.array(left_list, dtype=object),
            right_frame_indices=np.array(right_idx_list, dtype=object),
            left_frame_indices=np.array(left_idx_list, dtype=object),
            words=np.array(words, dtype=object),
        )

        print(f"saved {save_path} ({len(data)})")

    save("train", train)
    save("validation", val)
    save("test", test)

    print("\n===== STATS =====")
    for k, v in all_stats.items():
        print(
            k,
            "| total:", v["total_meta"],
            "| no_folder:", v["skip_no_folder"],
            "| no_word:", v["skip_no_word"],
            "| no_hands:", v["skip_no_hands"],
            "| kept:", v["kept"],
        )

    print("\nFinal counts")
    print("train:", len(train))
    print("validation:", len(val))
    print("test:", len(test))


if __name__ == "__main__":
    main()