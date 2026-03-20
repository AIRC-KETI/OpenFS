import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

def normalize(hand_pose, normalize_value=0.5):
    # hand_pose -= hand_pose[:, 0:1]

    # Scale normalize
    # max_vals = np.max(np.abs(hand_pose)) # Frame-by-Frame and Joint-by-Joint
    # hand_pose /= max_vals

    min_points = np.min(hand_pose, axis=1, keepdims=True)
    max_points = np.max(hand_pose, axis=1, keepdims=True)
    center_points = (min_points+max_points)/2
    hand_pose = hand_pose - center_points # The center point is at an origin
    hand_pose = hand_pose / (np.max(np.abs(hand_pose), axis=1, keepdims=True)+1e-6) # -1~1
    hand_pose = hand_pose * normalize_value # -0.5~0.5
    return hand_pose

def normalize_torch(hand_pose, normalize_value=0.5):
    # hand_pose: (B, T, J, D)
    # Calculate min/max for the J-axis (dim=2) (T remains unchanged)

    min_points, _ = torch.min(hand_pose, dim=2, keepdim=True)  # (B, T, 1, D)
    max_points, _ = torch.max(hand_pose, dim=2, keepdim=True)  # (B, T, 1, D)
    center_points = (min_points + max_points) / 2              # (B, T, 1, D)

    hand_pose = hand_pose - center_points                      # (B, T, J, D)
    max_abs, _ = torch.max(torch.abs(hand_pose), dim=2, keepdim=True)  # (B, T, 1, D)
    hand_pose = hand_pose / (max_abs + 1e-6)                   # [-1, 1]
    hand_pose = hand_pose * normalize_value                    # [-0.5, 0.5]

    return hand_pose

def preprocess_word(word, vocab):
    word = str(word).lower()

    syllable_list = []

    for w in word:
        syllable_list.append(w)
    word_indices = [len(vocab)] + convert_syllables_to_indices(syllable_list, vocab) + [0]

    return syllable_list, word_indices

def convert_syllables_to_indices(syllable_list, vocab):
    word_indices = []

    for w in syllable_list:
        word_indices.append(vocab[w])

    return word_indices

def make_frame_labels(word_indices, num_frames, vocab):
    word_indices = word_indices[1:-1]
    frame_labels = []
    for frame_idx, num_frame in enumerate(num_frames):
        if frame_idx % 2 == 0:
            frame_labels += [0] * num_frame
        else:
            frame_labels += [word_indices[frame_idx//2]] * num_frame
    return frame_labels

def make_frame_labels_wo_zero(word_indices, num_frames, vocab):
    word_indices = word_indices[1:-1]
    frame_labels = []
    for word_idx, num_frame in zip(word_indices, num_frames):
        frame_labels += [word_idx] * num_frame
    return frame_labels

def resample_pose(pose, rate_range=(0.5, 1.5)):
    rate = np.random.uniform(*rate_range)
    pose_resampled = zoom(pose, (rate, 1, 1), order=1)  # Linear interpolation
    return pose_resampled

def resample_pose_torch(pose, rate_range=(0.5, 1.5)):
    rate = np.random.uniform(*rate_range)
    original_length = pose.shape[0]
    num_features = pose.shape[1]
    dim_size = pose.shape[2]

    target_length = int(original_length * rate)
    pose_reshaped = pose.reshape(original_length, -1).unsqueeze(0).permute(0, 2, 1)
    pose_resampled = F.interpolate(
        pose_reshaped,
        size=target_length,
        mode='linear',
        align_corners=False # Align_corners should be False for interpolation on feature maps
    )
    pose_resampled = pose_resampled.permute(0, 2, 1).squeeze(0).reshape(target_length, num_features, dim_size)
    return pose_resampled

def resample_pose_and_label(pose, frame_label, rate_range=(0.5, 1.5)):
    rate = np.random.uniform(*rate_range)
    pose_resampled = zoom(pose, (rate, 1, 1), order=1)  # Linear interpolation
    frame_label_resampled = zoom(frame_label, (rate,), order=0)
    return pose_resampled, frame_label_resampled

def affine_pose(
    pose, scale_range=(0.8, 1.2), shear_range=(-0.2, 0.2),
    degree_range=(-30, 30), shift_range=(-0.1, 0.1)
):
    T, J, _ = pose.shape
    pose_affine = pose.copy()

    scale = np.diag(np.random.uniform(*scale_range, size=3)).astype(np.float32)  # (3, 3)

    shear = np.eye(3).astype(np.float32)
    shear[0,1] = np.random.uniform(*shear_range)
    shear[0,2] = np.random.uniform(*shear_range)
    shear[1,2] = np.random.uniform(*shear_range)

    angles = np.radians(np.random.uniform(*degree_range, size=3))
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]),  np.cos(angles[0])]
    ]).astype(np.float32)
    Ry = np.array([
        [ np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]).astype(np.float32)
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]),  np.cos(angles[2]), 0],
        [0, 0, 1]
    ]).astype(np.float32)
    R = Rz @ Ry @ Rx

    A = R @ shear @ scale

    t = np.random.uniform(*shift_range, size=(1, 1, 3)).astype(np.float32)

    pose_affine = np.einsum('tjc,cd->tjd', pose_affine, A) + t
    return pose_affine

def affine_pose_torch(
    pose: torch.Tensor,
    scale_range=(-0.8, 1.2),  # Adjusted default to be more sensible for scale
    shear_range=(-0.2, 0.2),
    degree_range=(-30, 30),
    shift_range=(-0.1, 0.1)
) -> torch.Tensor:
    """
    Applies a random affine transformation to a pose tensor using PyTorch.

    Args:
        pose (torch.Tensor): Input pose tensor of shape (T, J, C),
                             where T is time/sequence length, J is number of joints, C is coordinates (e.g., 3 for x,y,z).
        scale_range (tuple): Range for random scaling factors (min, max).
        shear_range (tuple): Range for random shear factors (min, max).
        degree_range (tuple): Range for random rotation angles in degrees (min, max).
        shift_range (tuple): Range for random translation factors (min, max).

    Returns:
        torch.Tensor: Affine-transformed pose tensor.
    """
    # Ensure pose is on the correct device if it's from a model output
    pose_affine = pose.clone()
    device = pose_affine.device

    # Identity matrices for transformations
    I_3x3 = torch.eye(3, device=device)

    # 1. Scale Matrix
    # Use torch.rand for random uniform values
    scale_factors = torch.rand(3, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale_mat = torch.diag(scale_factors) # (3, 3)

    # 2. Shear Matrix
    shear_mat = I_3x3.clone()
    # Fill upper triangle for shear (xy, xz, yz)
    shear_mat[0, 1] = torch.rand(1, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0] # xy shear
    shear_mat[0, 2] = torch.rand(1, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0] # xz shear
    shear_mat[1, 2] = torch.rand(1, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0] # yz shear

    # 3. Rotation Matrix
    angles = torch.rand(3, device=device) * (degree_range[1] - degree_range[0]) + degree_range[0]
    angles_rad = torch.deg2rad(angles)

    cx, cy, cz = torch.cos(angles_rad)
    sx, sy, sz = torch.sin(angles_rad)

    # Rotation matrix around X axis
    Rx = torch.tensor([
        [1,  0,  0],
        [0, cx, -sx],
        [0, sx,  cx]
    ], dtype=torch.float32, device=device)

    # Rotation matrix around Y axis
    Ry = torch.tensor([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ], dtype=torch.float32, device=device)

    # Rotation matrix around Z axis
    Rz = torch.tensor([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0, 1]
    ], dtype=torch.float32, device=device)

    R = Rz @ Ry @ Rx

    # Combine affine matrices
    A = R @ shear_mat @ scale_mat

    # 4. Translation vector
    # Using torch.rand with size=3 for 3D translation
    t = torch.rand(1, 1, 3, device=device) * (shift_range[1] - shift_range[0]) + shift_range[0] # (1, 1, 3)

    # Apply affine transformation: pose_transformed = pose @ A + t
    # For batch matrix multiplication: (T, J, C) * (C, C) -> (T, J, C)
    pose_affine = torch.einsum('tjc,cd->tjd', pose_affine, A) + t

    return pose_affine

def temporal_mask(pose, size_range=(0.2, 0.4), value=0.0):
    T = pose.shape[0]
    size = int(np.random.uniform(*size_range) * T)
    start = np.random.randint(0, T - size)
    pose[start:start+size] = value
    return pose

def temporal_mask_torch(pose: torch.Tensor, size_range=(0.2, 0.4), value=0.0) -> torch.Tensor:
    """
    Applies a temporal mask (zero-out a continuous time segment) to a pose tensor.

    Args:
        pose (torch.Tensor): Input pose tensor, e.g., (T, J, C) or (T, F).
        size_range (tuple): Range (min, max) for the mask size as a fraction of total time steps.
        value (float): Value to set masked elements to (default is 0.0).

    Returns:
        torch.Tensor: Pose tensor with a temporal mask applied.
    """
    # Create a copy to avoid in-place modification of original tensor
    pose_masked = pose.clone()

    T = pose_masked.shape[0] # Sequence length

    # Calculate mask size
    # torch.rand generates float in [0, 1), then scale and shift
    size_frac = torch.rand(1).item() * (size_range[1] - size_range[0]) + size_range[0]
    size = int(size_frac * T)

    if size >= T: # Handle cases where size might be equal to or greater than T
        return pose_masked.fill_(value)

    # Choose a random start index
    start = torch.randint(0, T - size + 1, (1,)).item()

    # Apply mask
    pose_masked[start : start + size] = value

    return pose_masked

def spatial_mask(pose, size_range=(0.05, 0.1), value=0.0):
    J = pose.shape[1]
    mask_size = int(np.random.uniform(*size_range) * J)
    joint_indices = np.random.choice(J, mask_size, replace=False)
    pose[:, joint_indices] = value
    return pose

def spatial_mask_torch(pose: torch.Tensor, size_range=(0.05, 0.1), value=0.0) -> torch.Tensor:
    """
    Applies a spatial mask (zero-out specific joints/features across all time steps) to a pose tensor.

    Args:
        pose (torch.Tensor): Input pose tensor, e.g., (T, J, C) or (T, F).
        size_range (tuple): Range (min, max) for the mask size as a fraction of total joints/features.
        value (float): Value to set masked elements to (default is 0.0).

    Returns:
        torch.Tensor: Pose tensor with a spatial mask applied.
    """
    # Create a copy to avoid in-place modification of original tensor
    pose_masked = pose.clone()

    J = pose_masked.shape[1] # Number of joints/features

    # Calculate mask size
    size_frac = torch.rand(1).item() * (size_range[1] - size_range[0]) + size_range[0]
    mask_size = int(size_frac * J)

    if mask_size == 0: # Avoid issues if mask_size rounds down to 0
        return pose_masked
    if mask_size >= J: # If all joints are masked
        return pose_masked.fill_(value)

    # Randomly select joint indices to mask without replacement
    # torch.randperm generates a permutation, then take the first mask_size elements
    joint_indices = torch.randperm(J, device=pose_masked.device)[:mask_size]

    # Apply mask across all time steps for selected joints
    pose_masked[:, joint_indices] = value

    return pose_masked