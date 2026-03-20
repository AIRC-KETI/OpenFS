import os, os.path as osp
import numpy as np
from collections import Counter
import glob
import time
from omegaconf import ListConfig

from torch.utils.data import Dataset

from lib.utils.dataset import (
    normalize, preprocess_word,
    make_frame_labels, resample_pose,
    affine_pose, temporal_mask, spatial_mask,
    make_frame_labels_wo_zero,
)


class MediaPipeDataset(Dataset):
    def __init__(
        self,
        data_path,
        vocab,
        representation='2d',
        return_video_name=False,
        do_augm=False,
        do_reverse=False,
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.vocab = vocab
        self.representation = representation
        self.return_video_name = return_video_name
        self.do_augm = do_augm
        self.do_reverse = do_reverse

        start_time = time.time()

        if isinstance(data_path, (list, ListConfig)):
            video_names, right_hand_poses, left_hand_poses, words, img_hw = [], [], [], [], []
            for _data_path in data_path:
                data = np.load(_data_path, allow_pickle=True)
                video_names.append(data["vidnames"])
                right_hand_poses.append(data["right_hand_poses"])
                left_hand_poses.append(data["left_hand_poses"])
                words.append(data["words"])
                img_hw.append(data["img_hw"])

            self.video_names = np.concatenate(video_names, axis=0)
            self.right_hand_poses = np.concatenate(right_hand_poses, axis=0)
            self.left_hand_poses = np.concatenate(left_hand_poses, axis=0)
            self.words = np.concatenate(words, axis=0)

        else:
            data = np.load(data_path, allow_pickle=True)
            print("Reading time:", time.time()-start_time)

            self.video_names = data["vidnames"]
            self.right_hand_poses = data["right_hand_poses"] # right_hand_poses
            self.left_hand_poses = data["left_hand_poses"] # left_hand_poses
            self.words = data["words"]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        item = {}

        if self.return_video_name:
            item['video_name'] = self.video_names[index]

        word = self.words[index].strip()

        if self.do_reverse:
            reverse_p = np.random.rand()
            if reverse_p < 0.3:
                do_reverse = True
            else:
                do_reverse = False
        else:
            do_reverse = False

        hand_swap_p = np.random.rand()
        if hand_swap_p < 0.5:
            do_hand_swap = True
        else:
            do_hand_swap = False

        syllable_list, syllable_indices = preprocess_word(word, self.vocab)
        syllable_indices = np.array(syllable_indices)
        if self.do_augm and do_reverse:
            word = word[::-1]
            syllable_indices[1:-1] = np.flip(syllable_indices[1:-1])
        item['word'] = word
        item['target_ids'] = syllable_indices

        poses_list = []
        seg_list   = []
        frame_list = []

        right_dict = self.right_hand_poses[index]  # person_id → (T_i, J, 3)
        left_dict  = self.left_hand_poses[index]

        # possible person_id
        all_ids = list(set(right_dict.keys()) | set(left_dict.keys()))
        if self.do_augm:
            np.random.shuffle(all_ids)
        else:
            all_ids.sort()

        # Repeat per person
        person_cnt = 0
        for i, person_id in enumerate(all_ids):
            if person_id in right_dict:
                right = np.array(right_dict[person_id], dtype=np.float32)
            else:
                right = np.empty((0, 21, 3))
            if person_id in left_dict:
                left  = np.array(left_dict[person_id],  dtype=np.float32)
            else:
                left = np.empty((0, 21, 3))

            # hand-swap augmentation
            if self.do_augm and do_hand_swap:
                right, left = left, right

            # Flip (reverse) augmentation
            if self.do_augm and do_reverse:
                if len(right) > 0:
                    right = np.flip(right, axis=0)
                if len(left) > 0:
                    left  = np.flip(left, axis=0)

            # normalize
            if len(right) > 0:
                # right = resize_and_pad_joints(right, img_h, img_w)
                right[..., :2] = normalize(right[..., :2], normalize_value=0.5)
                right[:, :, 2] -= right[:, 9:10, 2]

            if len(left) > 0:
                # left = resize_and_pad_joints(left, img_h, img_w)
                left[..., :2]  = normalize(left[..., :2], normalize_value=0.5)
                left[:, :, 2] -= left[:, 9:10, 2]

            # rotation augmentation
            if self.do_augm:
                if len(right) > 0:
                    if np.random.rand() < 0.8:
                        right = resample_pose(right)
                    if np.random.rand() < 0.75:
                        right = affine_pose(right)
                    if np.random.rand() < 0.5:
                        right = temporal_mask(right)
                    if np.random.rand() < 0.5:
                        right = spatial_mask(right)
                    # right = random_rotate_pose_3d(right)
                if len(left) > 0:
                    if np.random.rand() < 0.8:
                        left = resample_pose(left)
                    if np.random.rand() < 0.75:
                        left = affine_pose(left)
                    if np.random.rand() < 0.5:
                        left = temporal_mask(left)
                    if np.random.rand() < 0.5:
                        left = spatial_mask(left)
                    # left  = random_rotate_pose_3d(left)

            # segmentation values for person i: Right → 2*i, Left → 2*i+1
            seg_r = np.full(len(right), 2*person_cnt,   dtype=np.int64)
            seg_l = np.full(len(left),  2*person_cnt+1, dtype=np.int64)

            # frame_idx start from 0 for each person
            idx_r = np.arange(len(right), dtype=np.int64)
            idx_l = np.arange(len(left),  dtype=np.int64)

            if len(right) > 0:
                if self.representation == '2d':
                    right = right[..., :2]
                poses_list.append(right)
                seg_list.append(seg_r)
                frame_list.append(idx_r)
            if len(left) > 0:
                if self.representation == '2d':
                    left = left[..., :2]
                poses_list.append(left)
                seg_list.append(seg_l)
                frame_list.append(idx_l)
            if len(right) > 0 or len(left) > 0:
                person_cnt += 1
        item['poses'] = np.concatenate(poses_list,   axis=0)   # (T1+…+Tn, J, 3)
        item['rlh_seg'] = np.concatenate(seg_list,    axis=0)   # T_total
        item['frame_idx'] = np.concatenate(frame_list,  axis=0)   # T_total
        return item


class DGDataset(Dataset): # Diffusion-Generated Dataset
    def __init__(
        self,
        data_paths,
        vocab,
        representation='2d',
        do_augm=False,
        do_reverse=False,
        **kwargs
    ):
        super().__init__()

        self.data_paths = data_paths
        self.vocab = vocab
        self.representation = representation
        self.do_augm = do_augm
        self.do_reverse = do_reverse

        self.hand_poses = []
        self.words = []

        start_time = time.time()
        for data_path in data_paths:
            data = np.load(data_path, allow_pickle=True)
            print("Reading time:", time.time()-start_time)

            self.hand_poses.append(data["hand_poses"])
            self.words.append(data["words"])
        self.hand_poses = np.concatenate(self.hand_poses)
        self.words = np.concatenate(self.words)

        print('DGData length', len(self.words))

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        item = {}

        word = self.words[index].strip()

        do_reverse = self.do_reverse and np.random.rand() < 0.3
        do_hand_swap = self.do_augm and np.random.rand() < 0.5

        syllable_list, syllable_indices = preprocess_word(word, self.vocab)
        syllable_indices = np.array(syllable_indices)
        if self.do_augm and do_reverse:
            word = word[::-1]
            syllable_indices[1:-1] = np.flip(syllable_indices[1:-1])
        item['word'] = word
        item['target_ids'] = syllable_indices

        poses_list = []
        seg_list = []
        frame_list = []

        hand_poses = np.array(self.hand_poses[index], dtype=np.float32)  # (T, J, 3)
        if self.do_augm and do_reverse:
            hand_poses = np.flip(hand_poses, axis=0)

        hand_poses[..., :2] = normalize(hand_poses[..., :2], normalize_value=0.5)
        hand_poses[:, :, 2] -= hand_poses[:, 9:10, 2]

        synth_hand = np.mean(hand_poses, axis=0, keepdims=True).repeat(len(hand_poses), axis=0)
        T, J, C = synth_hand.shape
        temporal_noise = np.random.normal(scale=0.008, size=(T, 1, 1)).astype(np.float32)
        synth_hand += temporal_noise  # shape becomes (T, J, C) after broadcast

        if self.do_augm:
            if np.random.rand() < 0.8:
                hand_poses = resample_pose(hand_poses)
            if np.random.rand() < 0.75:
                hand_poses = affine_pose(hand_poses)
            if np.random.rand() < 0.5:
                hand_poses = temporal_mask(hand_poses)
            if np.random.rand() < 0.5:
                hand_poses = spatial_mask(hand_poses)

            if np.random.rand() < 0.8:
                synth_hand = resample_pose(synth_hand)
            if np.random.rand() < 0.75:
                synth_hand = affine_pose(synth_hand)
            if np.random.rand() < 0.5:
                synth_hand = temporal_mask(synth_hand)
            if np.random.rand() < 0.5:
                synth_hand = spatial_mask(synth_hand)

        # representation
        if self.representation == '2d':
            hand_poses = hand_poses[..., :2]
            synth_hand = synth_hand[..., :2]

        if do_hand_swap:
            hand_poses, synth_hand = synth_hand, hand_poses
        poses_list.extend([hand_poses, synth_hand])

        T_hand = hand_poses.shape[0]
        T_synth = synth_hand.shape[0]
        seg_list.extend([
            np.full(T_hand, 0, dtype=np.int64),  # real → seg 0
            np.full(T_synth, 1, dtype=np.int64)   # synth → seg 1
        ])
        frame_list.extend([
            np.arange(T_hand, dtype=np.int64),
            np.arange(T_synth, dtype=np.int64)
        ])

        item['poses'] = np.concatenate(poses_list,   axis=0)   # (T1+…+Tn, J, 3)
        item['rlh_seg'] = np.concatenate(seg_list,    axis=0)   # T_total
        item['frame_idx'] = np.concatenate(frame_list,  axis=0)   # T_total
        return item


class WordDataset(Dataset):
    def __init__(
        self,
        vocab,
        text_paths=None,
        min_frames_num=1,
        max_frames_num=3,
        min_transition_frames_num=1,
        max_transition_frames_num=3,
        max_word_length=None,
        min_word_length=None,
        num_samples=None,
        do_weight_length=False,
        do_rand_select=False,
        use_frame_label_clean=False,
        trim_input_ids=False,
        **kwargs
    ):
        super().__init__()

        self.vocab = vocab
        self.text_paths = text_paths
        self.min_frames_num = min_frames_num
        self.max_frames_num = max_frames_num
        self.min_transition_frames_num = min_transition_frames_num
        self.max_transition_frames_num = max_transition_frames_num
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length
        self.num_samples = num_samples
        self.do_weight_length = do_weight_length
        self.do_rand_select = do_rand_select
        self.use_frame_label_clean = use_frame_label_clean
        self.trim_input_ids = trim_input_ids

        ext_word_list = []
        for text_path in text_paths:
            if osp.isfile(text_path):
                with open(text_path, "r") as f:
                    ext_word_list += f.readlines()
            elif osp.isdir(text_path):
                text_path_list = os.listdir(text_path)
                text_path_list.sort()
                for base_path in text_path_list:
                    with open(osp.join(text_path, base_path), "r") as f:
                        ext_word_list += f.readlines()

        ext_word_list = list(set(ext_word_list))
        ext_word_list = [word.replace("\n", "") for word in ext_word_list]
        ext_word_list.sort()

        if max_word_length is not None:
            ext_word_list = [word for word in ext_word_list if len(word) <= max_word_length]
        if min_word_length is not None:
            ext_word_list = [word for word in ext_word_list if len(word) >= min_word_length]
        if num_samples is not None:
            ext_word_list = np.random.choice(ext_word_list, num_samples, replace=False)
        self.ext_word_list = ext_word_list
        print("len(self.ext_word_list):", len(self.ext_word_list))

        word_lengths = [len(word) for word in ext_word_list]
        length_counts = Counter(word_lengths)
        sample_weights = np.array([1.0 / length_counts[len(word)] for word in ext_word_list])
        self.sample_p = sample_weights / np.sum(sample_weights)


    def __len__(self):
        return len(self.ext_word_list)

    def __getitem__(self, index):
        item = {}

        if self.do_weight_length:
            ext_word = np.random.choice(self.ext_word_list, p=self.sample_p)
        elif self.do_rand_select:
            ext_word = np.random.choice(self.ext_word_list)
        else:
            ext_word = self.ext_word_list[index]
        item['word'] = ext_word

        ext_syllable_list, ext_syllable_indices = preprocess_word(ext_word, self.vocab)
        ext_syllable_indices = np.array(ext_syllable_indices)

        num_frames = []
        if self.use_frame_label_clean:
            for ext_syllable_idx in ext_syllable_indices[1:-1]:
                if ext_syllable_idx == 27:
                    num_frames.append(np.random.randint(self.min_transition_frames_num, self.max_transition_frames_num+1))
                else:
                    num_frames.append(np.random.randint(self.min_frames_num, self.max_frames_num+1))
            frame_labels = make_frame_labels_wo_zero(ext_syllable_indices, num_frames, self.vocab)
        else:
            num_frames.append(np.random.randint(self.min_transition_frames_num, self.max_transition_frames_num+1))
            num_syllables = len(ext_syllable_indices[1:-1])
            for _ in range(num_syllables):
                num_frames.append(np.random.randint(self.min_frames_num, self.max_frames_num+1))
                num_frames.append(np.random.randint(self.min_transition_frames_num, self.max_transition_frames_num+1))
            frame_labels = make_frame_labels(ext_syllable_indices, num_frames, self.vocab)

        item['frame_label_gen'] = frame_labels
        if self.trim_input_ids:
            item['target_ids_gen'] = ext_syllable_indices[1:-1]
        else:
            item['target_ids_gen'] = ext_syllable_indices
        item['frame_idx_gen'] = np.arange(len(frame_labels))
        return item