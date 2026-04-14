import sys
import os, os.path as osp
import argparse
import cv2
import mediapipe as mp
import numpy as np
import parmap
import time
import tqdm
import pandas as pd
from ultralytics import YOLO
import shutil
from collections import Counter

from lib.utils.bbox import uv2bbox, get_iou

detector = YOLO("data/yolo11s.pt")

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process(video_path, word, base_folder, debug=False):  # plus: ChicagoFSWildPlus
    if not debug:
        base_save_folder = osp.join(base_folder, 'mediapipe_results')
        save_folder = osp.join(base_save_folder, video_path)
        os.makedirs(save_folder, exist_ok=True)

        right_hand_poses_path = osp.join(save_folder, "right_hand_poses.npy")
        right_bbox_path = osp.join(save_folder, "right_hand_bbox.npy")
        right_frame_indices_path = osp.join(save_folder, "right_frame_indices.npy")
        left_hand_poses_path = osp.join(save_folder, "left_hand_poses.npy")
        left_bbox_path = osp.join(save_folder, "left_hand_bbox.npy")
        left_frame_indices_path = osp.join(save_folder, "left_frame_indices.npy")
        word_path = osp.join(save_folder, "word.npy")
        body_bbox_path = osp.join(save_folder, "bbox.npy")
        is_multi_path = osp.join(save_folder, "is_multi.npy")

        if \
            osp.exists(is_multi_path) and \
            (osp.exists(right_hand_poses_path) or osp.exists(left_hand_poses_path)) and \
            (osp.exists(right_bbox_path) or osp.exists(left_bbox_path)) and \
            (osp.exists(right_frame_indices_path) or osp.exists(left_frame_indices_path)) and \
            osp.exists(word_path):
            return False

    image_folder = osp.join(base_folder, 'video', video_path)
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(('.jpg', '.png'))
    ])

    bbox_results = detector.predict(image_files, classes=[0], conf=0.8, verbose=False)
    is_multi = np.mean([len(bbox_result) for bbox_result in bbox_results]) > 1.5
    if not debug:
        np.save(is_multi_path, is_multi)

    if is_multi:
        bbox_candidates = {}  # Dictionary to track each person, storing their bboxes per frame

        # YOLO sometimes fails to track and mixes person IDs,
        # so reorder based on previous bbox positions
        for bbox_result in bbox_results:
            # Extract frame index from filename
            frame_idx = int(osp.basename(bbox_result.path).replace(".jpg", ""))

            # Detected bounding boxes in xyxy format
            bboxes = bbox_result.boxes.xyxy.cpu().numpy().tolist()

            # Match each bbox with existing candidates
            for bbox in bboxes:
                is_succeed = False
                for candidate in bbox_candidates.values():
                    # Get the most recent bbox of this candidate
                    prev_key = sorted(candidate["bboxes"].keys())[-1]
                    prev_bbox = candidate["bboxes"][prev_key]

                    # Use IoU to determine same person
                    iou = get_iou(prev_bbox, bbox)
                    if iou > 0.8:
                        # Same person → update bbox
                        candidate["bboxes"][frame_idx] = bbox
                        is_succeed = True
                        break

                if not is_succeed:
                    # No match → create new candidate
                    bbox_candidates[len(bbox_candidates)+1] = {"bboxes": {frame_idx: bbox}}

            # If missing bbox in current frame → reuse previous bbox
            for candidate in bbox_candidates.values():
                bboxes = candidate['bboxes']
                if frame_idx not in bboxes:
                    prev_key = sorted(bboxes.keys())[-1]
                    prev_bbox = candidate["bboxes"][prev_key]
                    candidate["bboxes"][frame_idx] = prev_bbox

        # Fill missing frames using earliest bbox
        for candidate in bbox_candidates.values():
            bboxes = candidate["bboxes"]
            earliest_key = sorted(bboxes.keys())[0]
            for frame_idx in range(1, len(image_files)+1):
                if frame_idx not in bboxes:
                    earliest_bbox = candidate["bboxes"][earliest_key]
                    candidate["bboxes"][frame_idx] = earliest_bbox

        # Sort candidates by average position
        bbox_candidates = {
            new_id: data for new_id, (_, data) in enumerate(sorted(
                bbox_candidates.items(),
                key=lambda item: sum(
                    np.mean(
                        [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in item[1]['bboxes'].values()],
                        axis=0
                    ) if item[1]['bboxes'] else (0, 0)
                )
            ), start=1)
        }

        right_hand_pose_by_frame = {}
        left_hand_pose_by_frame = {}
        right_bbox_by_frame = {}
        left_bbox_by_frame = {}
        right_frame_indices = {}
        left_frame_indices = {}

        for person_id, candidate in bbox_candidates.items():
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                refine_face_landmarks=True,
            ) as holistic:
                for frame_idx, img_path in enumerate(image_files):
                    frame = cv2.imread(img_path)

                    bbox_body = candidate["bboxes"][frame_idx+1]
                    x1, y1, x2, y2 = map(int, bbox_body)
                    patch_body = frame[y1:y2, x1:x2]
                    patch_body.flags.writeable = False
                    patch_body_rgb = cv2.cvtColor(patch_body, cv2.COLOR_BGR2RGB)
                    results = holistic.process(patch_body_rgb)

                    if results.pose_landmarks is not None:
                        body_pose = np.array([
                            (landmark.x, landmark.y, landmark.z, landmark.visibility)
                            for landmark in results.pose_landmarks.landmark
                        ])

                        elbow_R, wrist_R = body_pose[14], body_pose[16]
                        img_h, img_w = patch_body.shape[:2]

                        # Process right hand
                        if elbow_R[1] > wrist_R[1]:
                            if results.right_hand_landmarks is not None:
                                right_hand_pose = np.array([
                                    [l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark
                                ])
                                right_hand_pose_uv = right_hand_pose[:, :2].copy()
                                right_hand_pose_uv[:, 0] *= img_w
                                right_hand_pose_uv[:, 1] *= img_h
                                right_hand_bbox = uv2bbox(right_hand_pose_uv)

                                if person_id not in right_hand_pose_by_frame:
                                    right_hand_pose_by_frame[person_id] = []
                                right_hand_pose_saved = np.concatenate(
                                    [right_hand_pose_uv, right_hand_pose[:, 2:]], axis=-1
                                )
                                right_hand_pose_by_frame[person_id].append(right_hand_pose_saved)

                                if person_id not in right_bbox_by_frame:
                                    right_bbox_by_frame[person_id] = []
                                right_bbox_by_frame[person_id].append(right_hand_bbox)

                                if person_id not in right_frame_indices:
                                    right_frame_indices[person_id] = []
                                right_frame_indices[person_id].append(frame_idx)

                        # Process left hand
                        elbow_L, wrist_L = body_pose[13], body_pose[15]
                        if elbow_L[1] > wrist_L[1]:
                            if results.left_hand_landmarks is not None:
                                left_hand_pose = np.array([
                                    [l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark
                                ])
                                left_hand_pose_uv = left_hand_pose[:, :2].copy()
                                left_hand_pose_uv[:, 0] *= img_w
                                left_hand_pose_uv[:, 1] *= img_h
                                left_hand_bbox = uv2bbox(left_hand_pose_uv)

                                if person_id not in left_hand_pose_by_frame:
                                    left_hand_pose_by_frame[person_id] = []
                                left_hand_pose_saved = np.concatenate(
                                    [left_hand_pose_uv, left_hand_pose[:, 2:]], axis=-1
                                )
                                left_hand_pose_by_frame[person_id].append(left_hand_pose_saved)

                                if person_id not in left_bbox_by_frame:
                                    left_bbox_by_frame[person_id] = []
                                left_bbox_by_frame[person_id].append(left_hand_bbox)

                                if person_id not in left_frame_indices:
                                    left_frame_indices[person_id] = []
                                left_frame_indices[person_id].append(frame_idx)

        if not debug:
            for person_id in right_hand_pose_by_frame:
                basename, ext = osp.splitext(osp.basename(right_hand_poses_path))
                path = osp.join(osp.dirname(right_hand_poses_path), f"{basename}_{person_id}{ext}")
                np.save(path, np.array(right_hand_pose_by_frame[person_id]))

            for person_id in left_hand_pose_by_frame:
                basename, ext = osp.splitext(osp.basename(left_hand_poses_path))
                path = osp.join(osp.dirname(left_hand_poses_path), f"{basename}_{person_id}{ext}")
                np.save(path, np.array(left_hand_pose_by_frame[person_id]))

            for person_id in bbox_candidates:
                basename, ext = osp.splitext(osp.basename(body_bbox_path))
                path = osp.join(osp.dirname(body_bbox_path), f"{basename}_{person_id}{ext}")
                np.save(path, bbox_candidates[person_id])

            np.save(word_path, word)

    else:
        right_hand_pose_by_frame = []
        left_hand_pose_by_frame = []

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
        ) as holistic:
            for frame_idx, img_path in enumerate(image_files):
                frame = cv2.imread(img_path)

                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                if results.pose_landmarks is not None:
                    body_pose = np.array([
                        (landmark.x, landmark.y, landmark.z, landmark.visibility)
                        for landmark in results.pose_landmarks.landmark
                    ])

                    elbow_R, wrist_R = body_pose[14], body_pose[16]
                    img_h, img_w = frame.shape[:2]

                    if elbow_R[1] > wrist_R[1]:
                        if results.right_hand_landmarks is not None:
                            right_hand_pose = np.array([
                                [l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark
                            ])
                            right_hand_pose_by_frame.append(right_hand_pose)

                    elbow_L, wrist_L = body_pose[13], body_pose[15]
                    if elbow_L[1] > wrist_L[1]:
                        if results.left_hand_landmarks is not None:
                            left_hand_pose = np.array([
                                [l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark
                            ])
                            left_hand_pose_by_frame.append(left_hand_pose)

        if not debug:
            np.save(right_hand_poses_path, np.array(right_hand_pose_by_frame))
            np.save(left_hand_poses_path, np.array(left_hand_pose_by_frame))
            np.save(word_path, word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plus', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_video_index', type=str)
    args = parser.parse_args()

    start_time = time.time()

    if args.plus:
        csv_path = 'data/raw/ChicagoFSWildPlus/ChicagoFSWildPlus.csv'
        base_folder = 'data/raw/ChicagoFSWildPlus'
    else:
        csv_path = 'data/raw/ChicagoFSWild/ChicagoFSWild.csv'
        base_folder = 'data/raw/ChicagoFSWild'

    base_save_folder = osp.join(base_folder, 'mediapipe_results')
    df_fs = pd.read_csv(csv_path)
    video_paths = df_fs["filename"].to_numpy()
    words = df_fs["label_proc"].to_numpy()

    if args.debug:
        for video_path, word in zip(video_paths, words):
            if args.debug_video_index in video_path:
                process(video_path, word, base_folder, debug=True)
    else:
        new_video_paths, new_words, new_base_folder = [], [], []
        for video_path, word in zip(video_paths, words):
            save_folder = osp.join(base_save_folder, video_path)
            word_path = osp.join(save_folder, "word.npy")

            if osp.exists(word_path):
                continue

            new_video_paths.append(video_path)
            new_words.append(word)
            new_base_folder.append(base_folder)

        print("len new_video_paths", len(new_video_paths))
        parmap.starmap(
            process,
            list(zip(new_video_paths, new_words, new_base_folder)),
            pm_pbar=True,
            pm_processes=2
        )
        print(time.time() - start_time)