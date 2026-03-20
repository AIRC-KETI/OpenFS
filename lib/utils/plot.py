import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os, os.path as osp
import imageio

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9,10), (10,11), (11,12),    # Middle
    (0,13), (13,14), (14,15), (15,16),   # Ring
    (0,17), (17,18), (18,19), (19,20)    # Pinky
]

def plot_pose_comparison(pose1=None, pose2=None, title="", frame_idx=0, bones=None):
    poses = [pose1, pose2]
    labels = ["Pose1", "Pose2"]
    colors = {"Pose1": "blue", "Pose2": "red"}

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, pose, label in zip(axes, poses, labels):
        if pose is not None:
            ax.scatter(pose[:, 0], pose[:, 1], c=colors[label])
            if bones is not None:
                for i, j in bones:
                    ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], c=colors[label], linewidth=2)
        ax.set_title(f"{label} - Frame {frame_idx}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.invert_yaxis()

    fig.suptitle(title)
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)[..., :3]
    plt.close(fig)
    return image

def save_pose_comparison_video(pose_seq1, pose_seq2=None, title="", save_path="", fps=10):
    images = []
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    n_frames = max(
        pose_seq1.shape[0],
        pose_seq2.shape[0] if pose_seq2 is not None else 0
    )

    for t in range(n_frames):
        frame1 = pose_seq1[t, :, :2] if t < pose_seq1.shape[0] else pose_seq1[-1, :, :2]
        frame2 = pose_seq2[t, :, :2] if (pose_seq2 is not None and t < pose_seq2.shape[0]) else (
            pose_seq2[-1, :, :2] if pose_seq2 is not None else None
        )
        image = plot_pose_comparison(frame1, frame2, title, t, BONES)

        h, w = image.shape[:2]
        new_w = (w + 15) // 16 * 16
        new_h = (h + 15) // 16 * 16
        image = cv2.resize(image, (new_w, new_h))
        images.append(image)

    imageio.mimsave(save_path, images, fps=fps)

def save_pose_comparison_image(pose1, pose2=None, title="", save_path=""):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    image = plot_pose_comparison(pose1[0, :, :2], pose2[0, :, :2] if pose2 is not None else None, title, 0, BONES)
    cv2.imwrite(save_path, image)

def rgb(r, g, b):
    return (r/255.0, g/255.0, b/255.0)

FINGER_COLORS = [
    rgb(255, 140,   0),  # Thumb - vivid orange
    rgb(128,   0, 128),  # Index - pure purple
    rgb(255, 215,   0),  # Middle - bright yellow
    rgb(  0, 255,   0),  # Ring - pure green
    rgb(  0,   0, 255),  # Pinky - pure blue
]


def draw_hand_2d(ax, joints_xy):
    # bones per finger (5 fingers × 4 bones)
    for f_idx in range(5):
        color = FINGER_COLORS[f_idx]
        for b in range(4):
            i, j = BONES[f_idx * 4 + b]
            ax.plot(
                [joints_xy[i, 0], joints_xy[j, 0]],
                [joints_xy[i, 1], joints_xy[j, 1]],
                c=color, linewidth=2
            )
        # scatter 4 fingertip joints + base (for each finger)
        finger_joint_ids = [BONES[f_idx * 4 + b][1] for b in range(4)]
        ax.scatter(
            joints_xy[finger_joint_ids, 0],
            joints_xy[finger_joint_ids, 1],
            c=[color],
            s=20,
            edgecolors="k",
            linewidths=0.3
        )

    # wrist
    ax.scatter(
        joints_xy[0, 0],
        joints_xy[0, 1],
        c=[(0, 0, 0)],
        s=25,
        edgecolors="k"
    )

def render_frame_2d(joints_xy, save_path, label_str=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_hand_2d(ax, joints_xy)

    ax.set_aspect("equal")
    ax.axis("off")

    x_min, y_min = joints_xy.min(axis=0)
    x_max, y_max = joints_xy.max(axis=0)
    margin = 0.5 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(-0.5-margin, 0.5+margin)
    ax.set_ylim(0+margin, -1-margin)
    
    if label_str is not None:
        label_upper = str(label_str).upper()

        ax.text(
            0.02, 0.02, label_upper,
            transform=ax.transAxes,
            fontsize=20,
            color="black",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def render_sequence_2d(xyz_seq, save_dir, labels=None):
    os.makedirs(save_dir, exist_ok=True)

    T = xyz_seq.shape[0]
    for idx in range(T):
        joints_xy = xyz_seq[idx, :, :2]
        save_path = os.path.join(save_dir, f"index_{idx:04d}.png")

        label_str = None
        if labels is not None:
            if isinstance(labels, (list, tuple)):
                label_str = labels[idx]
            elif isinstance(labels, str):
                label_str = list(labels)[idx]
            else:
                label_str = str(labels)

        render_frame_2d(joints_xy, save_path, label_str=label_str)