import torch
import torch.nn as nn

from lib.utils.mask import get_batch_attention_mask
from lib.utils.pos import (
    positional_encoding_from_idx,
    build_frame_idx_ext_torch,
    build_frame_idx_ext_torch_feats,
)
import constants


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        char_size, pose_dim=21*3, d_model=512,
        nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        **kwargs
    ):
        super().__init__()

        self.char_size = char_size
        self.pose_dim = pose_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.time_embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pose_embedding = nn.Linear(pose_dim, d_model//2)
        self.char_embedding = nn.Embedding(char_size+1, d_model//2, padding_idx=char_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder_head = nn.Linear(d_model, pose_dim)

    def forward(self, timesteps, poses_noised, frame_label, frame_idx):
        """
        poses_noised: (B, L, 63)
        """
        B, L, J, D = poses_noised.shape
        device = poses_noised.device

        poses_noised = poses_noised.reshape(B, L, J*D)
        time_emb = self.time_embedding(positional_encoding_from_idx(timesteps.reshape(B, 1), self.d_model, device=device))

        frame_idx_enc = positional_encoding_from_idx(frame_idx, self.d_model, device=device)
        pose_emb = self.pose_embedding(poses_noised)
        char_emb = self.char_embedding(frame_label)
        src_emb = torch.cat([pose_emb, char_emb], dim=2) # channel-wise
        src_emb = src_emb + frame_idx_enc
        src_emb = torch.cat([time_emb, src_emb], dim=1) # frame-wise

        src_attn_mask = get_batch_attention_mask(frame_label, constants.MINUS_ONE_HUNDRED_VALUE)
        src_attn_mask = torch.cat([torch.zeros(B, 1, device=device), src_attn_mask], dim=1)
        encoded_features = self.encoder(src_emb, src_key_padding_mask=src_attn_mask)
        x0_pred = self.encoder_head(encoded_features)
        x0_pred = x0_pred[:, 1:] # remove time slot
        x0_pred = x0_pred.reshape(B, L, J, D)
        return {'output': x0_pred, 'encoded_features': encoded_features}


class EncoderTransformerGlobal(nn.Module):
    def __init__(
        self,
        char_size, pose_dim=21*3, d_model=512,
        nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        **kwargs
    ):
        super().__init__()

        self.char_size = char_size
        self.pose_dim = pose_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.time_embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pose_embedding = nn.Linear(pose_dim, d_model)
        self.char_embedding = nn.Embedding(char_size+1, d_model, padding_idx=char_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder_head = nn.Linear(d_model, pose_dim)

    def forward(self, timesteps, poses_noised, word_label, poses, frame_idx):
        """
        poses_noised: (B, L, 63)
        """
        B, L, J, D = poses_noised.shape
        device = poses_noised.device
        L_w = word_label.shape[1]

        poses_noised = poses_noised.reshape(B, L, J*D)
        time_emb = self.time_embedding(positional_encoding_from_idx(timesteps.reshape(B, 1), self.d_model, device=device))

        frame_idx = build_frame_idx_ext_torch(frame_idx, word_label, max_val=1024)
        frame_idx_enc = positional_encoding_from_idx(frame_idx, self.d_model, device=device)
        pose_emb = self.pose_embedding(poses_noised)
        char_emb = self.char_embedding(word_label)
        src_emb = torch.cat([char_emb, pose_emb], dim=1) # frame-wise
        src_emb = src_emb + frame_idx_enc
        src_emb = torch.cat([time_emb, src_emb], dim=1) # frame-wise

        src_attn_mask1 = get_batch_attention_mask(word_label, constants.MINUS_ONE_HUNDRED_VALUE)
        src_attn_mask2 = get_batch_attention_mask(poses, constants.MINUS_TWO_VALUE)
        src_attn_mask = torch.cat([torch.zeros(B, 1, device=device), src_attn_mask1, src_attn_mask2], dim=1)
        encoded_features = self.encoder(src_emb, src_key_padding_mask=src_attn_mask)
        x0_pred = self.encoder_head(encoded_features)
        x0_pred = x0_pred[:, 1+L_w:] # remove time slot + word slot
        x0_pred = x0_pred.reshape(B, L, J, D)
        return {'output': x0_pred, 'encoded_features': encoded_features}


class EncoderTransformerWord(nn.Module):
    def __init__(
        self,
        char_size, pose_dim=21*3, d_model=512,
        nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        **kwargs
    ):
        super().__init__()

        self.char_size = char_size
        self.pose_dim = pose_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.time_embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pose_embedding = nn.Linear(pose_dim, d_model)
        self.char_embedding = nn.Linear(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder_head = nn.Linear(d_model, pose_dim)

    def forward(self, timesteps, poses_noised, word_feats, poses, frame_idx):
        """
        poses_noised: (B, L, 63)
        """
        B, L, J, D = poses_noised.shape
        device = poses_noised.device
        word_feats = word_feats[:, None]
        L_w = word_feats.shape[1]

        poses_noised = poses_noised.reshape(B, L, J*D)
        time_emb = self.time_embedding(positional_encoding_from_idx(timesteps.reshape(B, 1), self.d_model, device=device))

        frame_idx = build_frame_idx_ext_torch_feats(frame_idx, word_feats, max_val=1024)
        frame_idx_enc = positional_encoding_from_idx(frame_idx, self.d_model, device=device)
        pose_emb = self.pose_embedding(poses_noised)
        char_emb = self.char_embedding(word_feats)
        src_emb = torch.cat([char_emb, pose_emb], dim=1) # frame-wise
        src_emb = src_emb + frame_idx_enc
        src_emb = torch.cat([time_emb, src_emb], dim=1) # frame-wise

        src_attn_mask1 = get_batch_attention_mask(word_feats, constants.MINUS_ONE_HUNDRED_VALUE)
        src_attn_mask2 = get_batch_attention_mask(poses, constants.MINUS_TWO_VALUE)
        src_attn_mask = torch.cat([torch.zeros(B, 1, device=device), src_attn_mask1, src_attn_mask2], dim=1)
        encoded_features = self.encoder(src_emb, src_key_padding_mask=src_attn_mask)
        x0_pred = self.encoder_head(encoded_features)
        x0_pred = x0_pred[:, 1+L_w:] # remove time slot + word slot
        x0_pred = x0_pred.reshape(B, L, J, D)
        return {'output': x0_pred, 'encoded_features': encoded_features}