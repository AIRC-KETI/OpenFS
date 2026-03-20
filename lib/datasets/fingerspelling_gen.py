import numpy as np
import time

from torch.utils.data import Dataset

from lib.utils.dataset import (
    normalize, preprocess_word,
    resample_pose, resample_pose_and_label,
    affine_pose,
)


class MediaPipeDataset(Dataset):
    def __init__(
        self,
        data_path, vocab,
        representation='2d',
        return_video_name=False,
        do_augm=False,
        do_affine=False,
        do_reverse=False,
        trim_input_ids=False,
        use_frame_label=False,
        is_both_hands=False,
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.vocab = vocab
        self.representation = representation
        self.return_video_name = return_video_name
        self.do_augm = do_augm
        self.do_affine = do_affine
        self.do_reverse = do_reverse
        self.trim_input_ids = trim_input_ids
        self.use_frame_label = use_frame_label
        self.is_both_hands = is_both_hands

        start_time = time.time()
        data = np.load(data_path, allow_pickle=True)
        print("Reading time:", time.time()-start_time)

        self.video_names = data["vidnames"]
        self.right_hand_poses = data["right_hand_poses"] # right_hand_poses
        self.left_hand_poses = data["left_hand_poses"] # left_hand_poses
        self.words = data["words"]
        self.img_hw = data["img_hw"]
        self.signer_id = data["signer_id"]
        self.is_multi = data["is_multi"]
        self.signing_hand = data["signing_hand"]
        if use_frame_label:
            if 'frame_label_clean' in data:
                self.frame_label_clean = data['frame_label_clean']
            else:
                self.frame_label_clean = None
            if 'frame_label' in data:
                self.frame_label = data['frame_label']
            else:
                raise Exception("Missing 'frame_label' in data.")
            self._proc()
            self._proc_frame_label()
            if self.frame_label_clean is not None:
                assert len(self.frame_label) == len(self.frame_label_clean), "Length mismatch: frame_label vs frame_label_clean."
                self.frame_label = self.frame_label_clean
                del self.frame_label_clean
        else:
            self._proc()
            self.frame_label = None

    def _proc(self):
        masks = []
        for i in range(len(self.signing_hand)):
            signing_hand = self.signing_hand[i]
            if signing_hand is not None:
                masks.append(i)
        self.video_names = self.video_names[masks]
        self.right_hand_poses = self.right_hand_poses[masks]
        self.left_hand_poses = self.left_hand_poses[masks]
        self.words = self.words[masks]
        self.img_hw = self.img_hw[masks]
        self.signer_id = self.signer_id[masks]
        self.is_multi = self.is_multi[masks]
        self.signing_hand = self.signing_hand[masks]
        if hasattr(self, 'frame_label'):
            self.frame_label = self.frame_label[masks]

    def _proc_frame_label(self):
        masks = []
        # masks2 = []
        for i in range(len(self.frame_label)):
            frame_label = self.frame_label[i]
            words = self.words[i]

            unique_frame_values, indices = np.unique(frame_label, return_index=True)
            sorted_indices = np.argsort(indices)
            ordered_frame_labels = unique_frame_values[sorted_indices]
            ordered_frame_labels = ordered_frame_labels[ordered_frame_labels != -1]

            words_list = list(words)
            word_ids = [self.vocab[word] for word in words_list if word in self.vocab]
            word_ids = np.array(word_ids)
            unique_word_values, word_indices = np.unique(word_ids, return_index=True)
            sorted_word_indices = np.argsort(word_indices)
            ordered_word_ids = unique_word_values[sorted_word_indices]

            are_equal = np.array_equal(ordered_frame_labels, ordered_word_ids)
            if are_equal:
                masks.append(i)
            # if len(np.unique(frame_label))-1 == len(np.unique(list(words))):
            #     masks2.append(i)
        self.video_names = self.video_names[masks]
        self.right_hand_poses = self.right_hand_poses[masks]
        self.left_hand_poses = self.left_hand_poses[masks]
        self.words = self.words[masks]
        self.img_hw = self.img_hw[masks]
        self.signer_id = self.signer_id[masks]
        self.is_multi = self.is_multi[masks]
        self.signing_hand = self.signing_hand[masks]
        self.frame_label = self.frame_label[masks]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        item = {}

        if self.return_video_name:
            item['video_name'] = self.video_names[index]

        word = self.words[index].strip()
        signing_hand = self.signing_hand[index]

        reverse_p = np.random.rand()

        if self.do_reverse:
            if reverse_p < 0.3: # 0.3
                do_reverse = True
            else:
                do_reverse = False
        else:
            do_reverse = False

        syllable_list, syllable_indices = preprocess_word(word, self.vocab)
        syllable_indices = np.array(syllable_indices)
        if self.do_augm and do_reverse:
            word = word[::-1]
            syllable_indices[1:-1] = np.flip(syllable_indices[1:-1])
        item['word'] = word

        if self.trim_input_ids:
            item['input_ids'] = syllable_indices[1:-1]
        else:
            item['input_ids'] = syllable_indices

        right_dict = self.right_hand_poses[index]  # person_id → (T_i, J, 3)
        left_dict  = self.left_hand_poses[index]

        # possible person_id
        is_multi = self.is_multi[index]
        all_ids = list(set(right_dict.keys()) | set(left_dict.keys()))
        all_ids.sort()

        if is_multi:
            person_id = all_ids[signing_hand//2]
        else:
            person_id = all_ids[all_ids.index(signing_hand//2)]

        if signing_hand % 2 == 0: # R hand
            poses = np.array(right_dict[person_id], dtype=np.float32)
        elif signing_hand % 2 == 1: # L hand
            poses = np.array(left_dict[person_id], dtype=np.float32)

        if self.frame_label is not None:
            frame_label = self.frame_label[index]
        else:
            frame_label = None

        # Flip (reverse) augmentation
        if self.do_augm and do_reverse:
            poses = np.flip(poses, axis=0)
            if frame_label is not None:
                frame_label = np.flip(frame_label, axis=0)

        # normalize
        poses[..., :2] = normalize(poses[..., :2], normalize_value=0.5)
        poses[:, 9:10, 2] -= poses[:, 9:10, 2]

        # rotation augmentation
        if self.do_augm:
            if np.random.rand() < 0.8:
                if frame_label is not None:
                    poses, frame_label = resample_pose_and_label(poses, frame_label)
                else:
                    poses = resample_pose(poses)
            if self.do_affine and np.random.rand() < 0.75:
                poses = affine_pose(poses)

        frame_idx = np.arange(len(poses), dtype=np.int64)

        if self.representation == '2d':
            poses = poses[..., :2]

        item['poses'] = poses
        if frame_label is not None:
            item['frame_label'] = frame_label
        item['frame_idx'] = frame_idx
        return item