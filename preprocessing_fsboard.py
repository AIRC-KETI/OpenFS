import os
import os.path as osp
import argparse
import cv2
import mediapipe as mp
import numpy as np
import json
import tqdm
import time
import parmap

from lib.utils.bbox import uv2bbox

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- add this near the top of process() in your first script ---
def _already_done(save_folder: str) -> bool:
    # Always require word.npy (so we don't skip half-written results)
    word_path = osp.join(save_folder, "word.npy")

    # Either hand side can exist (some clips genuinely have only one hand detected)
    right_ok = (
        osp.exists(osp.join(save_folder, "right_hand_poses.npy"))
        and osp.exists(osp.join(save_folder, "right_hand_bbox.npy"))
        and osp.exists(osp.join(save_folder, "right_frame_indices.npy"))
    )
    left_ok = (
        osp.exists(osp.join(save_folder, "left_hand_poses.npy"))
        and osp.exists(osp.join(save_folder, "left_hand_bbox.npy"))
        and osp.exists(osp.join(save_folder, "left_frame_indices.npy"))
    )

    return osp.exists(word_path) and (right_ok or left_ok)

def process(video_file, word, save_folder, debug=False):

    os.makedirs(save_folder, exist_ok=True)
    if _already_done(save_folder):
        return

    if (
        osp.exists(osp.join(save_folder, "right_hand_poses.npy"))
        and osp.exists(osp.join(save_folder, "left_hand_poses.npy"))
        and osp.exists(osp.join(save_folder, "word.npy"))
    ):
        return
    print(video_file)
    cap = cv2.VideoCapture(video_file)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    print(rotation)

    right_hand_pose_by_frame = []
    left_hand_pose_by_frame = []
    right_bbox_by_frame = []
    left_bbox_by_frame = []
    right_frame_indices = []
    left_frame_indices = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            MAX_SIZE = 640
            img_h, img_w = frame.shape[:2]
            scale = MAX_SIZE / max(img_h, img_w)
            frame.flags.writeable = False            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.right_hand_landmarks is not None:
                right_hand_pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                )
                right_hand_pose_uv = right_hand_pose[:, :2]
                right_hand_pose_uv[:, 0] *= img_w
                right_hand_pose_uv[:, 1] *= img_h

                right_hand_pose_by_frame.append(
                    np.concatenate([right_hand_pose_uv, right_hand_pose[:, 2:]], axis=-1)
                )
                right_bbox_by_frame.append(uv2bbox(right_hand_pose_uv))
                right_frame_indices.append(frame_idx)

            if results.left_hand_landmarks is not None:
                left_hand_pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                )
                left_hand_pose_uv = left_hand_pose[:, :2]
                left_hand_pose_uv[:, 0] *= img_w
                left_hand_pose_uv[:, 1] *= img_h

                left_hand_pose_by_frame.append(
                    np.concatenate([left_hand_pose_uv, left_hand_pose[:, 2:]], axis=-1)
                )
                left_bbox_by_frame.append(uv2bbox(left_hand_pose_uv))
                left_frame_indices.append(frame_idx)

            if debug:
                vis = frame.copy()
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(vis, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(vis, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # vis = cv2.resize(vis, (img_w//4, img_h//4))
                cv2.imshow("debug", vis)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            frame_idx += 1

    cap.release()
    if debug:
        cv2.destroyAllWindows()

    np.save(osp.join(save_folder, "right_hand_poses.npy"),
            np.array(right_hand_pose_by_frame, dtype=np.float32))
    np.save(osp.join(save_folder, "right_hand_bbox.npy"),
            np.array(right_bbox_by_frame, dtype=np.int16))
    np.save(osp.join(save_folder, "right_frame_indices.npy"),
            np.array(right_frame_indices, dtype=np.int16))

    np.save(osp.join(save_folder, "left_hand_poses.npy"),
            np.array(left_hand_pose_by_frame, dtype=np.float32))
    np.save(osp.join(save_folder, "left_hand_bbox.npy"),
            np.array(left_bbox_by_frame, dtype=np.int16))
    np.save(osp.join(save_folder, "left_frame_indices.npy"),
            np.array(left_frame_indices, dtype=np.int16))

    np.save(osp.join(save_folder, "word.npy"), word)


if __name__ == "__main__":
    BASE_DIR = "data/raw/FSboard"

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    start_time = time.time()

    dataset_splits = {
        "daun_v3": ["train", "validation", "test"],
        "dmk_v3": ["train"],
    }

    for dataset, splits in dataset_splits.items():
        jobs = []

        dataset_root = osp.join(BASE_DIR, dataset)

        for split in splits:

            metadata_path = osp.join(
                dataset_root,
                "metadata",
                f"{dataset}-{split}.json"
            )

            if not osp.exists(metadata_path):
                continue

            print(f"\nProcessing {dataset} - {split}")

            with open(metadata_path, "r") as f:
                meta = json.load(f)

            for item in meta:

                video_file = osp.join(
                    dataset_root,
                    "video_clips",
                    f"{dataset}-{split}",
                    item["clipFilename"]
                )

                save_folder = osp.join(
                    dataset_root,
                    "mediapipe_results",
                    split,
                    osp.splitext(item["clipFilename"])[0]
                )

                if osp.exists(osp.join(save_folder, "word.npy")):
                    continue

                if not osp.exists(video_file):
                    continue

                jobs.append((video_file, item["phrase"], save_folder))

        print("Total jobs:", len(jobs))
        cpu_count = os.cpu_count()
        pm_processes = max(1, cpu_count // 2)
        pm_processes = min(8, pm_processes)

        print("CPU:", cpu_count)
        print("Using processes:", pm_processes)
        parmap.starmap(
            process,
            jobs,
            pm_pbar=True,
            pm_processes=pm_processes
        )

    print("\nDone.", time.time() - start_time)