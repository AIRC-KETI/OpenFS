import os, os.path as osp
import argparse
import numpy as np
import pandas as pd
import tqdm
import glob

def main(task, plus):
    if plus:
        csv_path = "data/raw/ChicagoFSWildPlus/ChicagoFSWildPlus.csv"
        base_save_folder = "data/raw/ChicagoFSWildPlus/mediapipe_results"
        save_path = f"data/raw/ChicagoFSWildPlus/ChicagoFSWildPlus_{task}.npz"
    else:
        csv_path = "data/raw/ChicagoFSWild/ChicagoFSWild.csv"
        base_save_folder = "data/raw/ChicagoFSWild/mediapipe_results"
        save_path = f"data/raw/ChicagoFSWild/ChicagoFSWild_{task}.npz"

    df = pd.read_csv(csv_path)

    video_paths = df["filename"]
    label_procs = df["label_proc"]
    widths = df["width"]
    heights = df["height"]
    partitions = df["partition"]
    signer_ids = df["signer"]

    vid_paths = []
    right_hand_poses_list = []
    left_hand_poses_list = []
    word_list = []
    img_hw_list = []
    signer_id_list = []
    is_multi_list = []

    for video_path, word, width, height, partitions, signer_id in \
        tqdm.tqdm(zip(video_paths, label_procs, widths, heights, partitions, signer_ids), total=len(video_paths)):
        if partitions != task:
            continue
        if word is np.nan:
            continue
        save_folder = osp.join(base_save_folder, video_path)

        is_multi = np.load(osp.join(save_folder, 'is_multi.npy'))
        if is_multi:
            right_hand_poses = {}
            for path in glob.glob(osp.join(save_folder, "right_hand_poses_*.npy")):
                pid = int(osp.basename(path).split("_")[-1].split(".")[0])
                right_hand_poses[pid] = np.load(path)

            left_hand_poses = {}
            for path in glob.glob(osp.join(save_folder, "left_hand_poses_*.npy")):
                pid = int(osp.basename(path).split("_")[-1].split(".")[0])
                arr = np.load(path)
                if arr.ndim == 3 and arr.shape[-1] >= 1:
                    arr[..., 0] = 1.0 - arr[..., 0]
                left_hand_poses[pid] = arr

            img_hw = {}
            for path in glob.glob(osp.join(save_folder, "bbox_*.npy")):
                pid = int(osp.basename(path).split("_")[-1].split(".")[0])
                arrs = np.load(path, allow_pickle=True)
                img_hw[pid] = []
                for arr in arrs[()]['bboxes'].values():
                    x1, y1, x2, y2 = arr
                    _height, _width = int(y2-y1), int(x2-x1)
                    img_hw[pid].append([_height, _width])
        else:
            right_hand_poses = {0: np.load(osp.join(save_folder, "right_hand_poses.npy")) \
                if osp.exists(osp.join(save_folder, "right_hand_poses.npy")) else np.zeros((0,21,3))}
            
            # left hand coords to right hand coords
            left_arr = np.load(osp.join(save_folder, "left_hand_poses.npy")) \
                if osp.exists(osp.join(save_folder, "left_hand_poses.npy")) else np.zeros((0,21,3))
            if left_arr.ndim == 3 and left_arr.shape[-1] >= 1:
                left_arr[..., 0] = 1.0 - left_arr[..., 0]
            left_hand_poses = {0: left_arr}

            img_hw = {0: [height, width]}
        
        is_save = False
        for poses in right_hand_poses.values():
            if len(poses) > 0:
                is_save = True
        for poses in left_hand_poses.values():
            if len(poses) > 0:
                is_save = True

        if task=='test' or is_save:
            vid_paths.append(video_path)
            right_hand_poses_list.append(right_hand_poses)
            left_hand_poses_list.append(left_hand_poses)
            word = word.strip()
            word_list.append(word)
            img_hw_list.append(img_hw)
            signer_id_list.append(signer_id)
            is_multi_list.append(is_multi)

    np.savez(
        save_path, 
        vidnames=vid_paths, 
        right_hand_poses=np.array(right_hand_poses_list, dtype=object), 
        left_hand_poses=np.array(left_hand_poses_list, dtype=object), 
        words=np.array(word_list, dtype=object), 
        img_hw=np.array(img_hw_list), 
        signer_id=np.array(signer_id_list),
        is_multi=np.array(is_multi_list),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plus', action='store_true')
    args = parser.parse_args()

    main(task="train", plus=args.plus)
    main(task="dev", plus=args.plus)
    main(task="test", plus=args.plus)