import os
import numpy as np
from tqdm import tqdm

def make_npz(split='train', data_name='neobench', is_filter=False):
    if data_name == 'words_alpha':
        base_dir = f'data/generated/{split}/poses'
        suffix = ""
        if is_filter:
            suffix += "_filtered"
        out_path = f"data/fingerspelling/{data_name}_{split}{suffix}_gen.npz"
    elif data_name == "neobench":
        base_dir = f'data/generated/{data_name}/poses'
        out_path = f"data/fingerspelling/{data_name}_{split}_gen.npz"

    hand_poses = []
    words = []

    if is_filter:
        with open('data/english_word/except_words_alpha.txt', 'r') as f:
            except_words_alpha_list = f.read().splitlines()

    for word in tqdm(os.listdir(base_dir)):
        if is_filter and word in except_words_alpha_list:
            continue

        word_dir = os.path.join(base_dir, word)
        if not os.path.isdir(word_dir):
            continue

        for fname in sorted(os.listdir(word_dir)):
            if fname.endswith(".npy"):
                fpath = os.path.join(word_dir, fname)
                hand_poses.append(np.load(fpath))
                words.append(word)

    hand_poses = np.array(hand_poses, dtype=object)
    words = np.array(words)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, hand_poses=hand_poses, words=words)
    print(f"Saved {len(words)} samples to {out_path}")

if __name__ == "__main__":
    make_npz(split='test', data_name='neobench')
    make_npz(split='train', data_name='words_alpha', is_filter=True)