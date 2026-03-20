import torch
import torch.nn as nn

from lib.models.transformer import (
    TransformerDecoderLayer, TransformerDecoder
)
from lib.utils.mask import get_batch_attention_mask
from lib.utils.pos import sinusoidal_positional_encoding, positional_encoding_from_idx
import constants


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        char_size,
        input_dim=96, max_len=128,
        d_model=512, nhead=8,
        num_layers=6, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        conventional_positional_encoding=False,
        **kwargs
    ):
        super().__init__()

        self.char_size = char_size
        self.input_dim = input_dim
        self.max_len = max_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.cpe = conventional_positional_encoding # CPE: conventional_positional_encoding

        self.pose_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.char_embedding = nn.Embedding(char_size+1, d_model, padding_idx=char_size)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, char_size)
        )

    def forward(self, poses, rlh_seg, frame_idx, target_ids=None):
        """
        poses: (B, L, J, 6)       — pose + word + pad token IDs
        segmentation: (B, L)    — 0: pose, 1: word, 2: padding
        """
        B, L, J, pose_dim = poses.shape
        device = poses.device

        poses_reshaped = poses.reshape(B, L, J*pose_dim)
        pose_emb = self.pose_embedding(poses_reshaped)
        if self.cpe:
            pos_enc = sinusoidal_positional_encoding(L, self.d_model, device=device)
            pose_emb = pose_emb + pos_enc
        else:
            frame_idx_enc = positional_encoding_from_idx(frame_idx, self.d_model, device=device)
            pose_emb = pose_emb + frame_idx_enc

            hand_type_enc = positional_encoding_from_idx(rlh_seg, self.d_model, device=device)
            pose_emb = pose_emb + hand_type_enc

        src_attn_mask = get_batch_attention_mask(poses, constants.MINUS_TWO_VALUE)
        encoded_features = self.encoder(pose_emb, src_key_padding_mask=src_attn_mask)

        output = {}
        output['encoder_memory'] = encoded_features

        if target_ids is not None:
            L_char = target_ids.shape[1]
            char_emb = self.char_embedding(target_ids)
            char_enc = sinusoidal_positional_encoding(L_char, self.d_model, device=device)

            # Causal mask
            tgt_mask = torch.triu(torch.ones(L_char, L_char, device=device), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
            char_emb = char_emb + char_enc

            # Padding mask (B, L) — True where padding
            # tgt_key_padding_mask = (target_ids == self.char_size)
            tgt_key_padding_mask = get_batch_attention_mask(target_ids, self.char_size)

            # Decoder
            out, cross_attns = self.decoder(
                tgt=char_emb,
                memory=encoded_features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_attn_mask,
            )  # (B, L, d_model)

            logits = self.lm_head(out)  # (B, L, vocab_size)

            output['logits'] = logits
            output['cross_attns'] = cross_attns
            output['out_feat'] = out

        return output

    def generate(self, poses, rlh_seg, frame_idx, max_len=32, bos_token_id=32, eos_token_id=0, pad_token_id=None, pad_pos_idx=33):
        """
        pose_ids: (B, L)       — pose + word + pad token IDs
        rlh_seg: (B, L)        — 0: pose, 1: word, 2: padding
        """
        if pad_token_id is None:
            pad_token_id = self.char_size

        B, L, J, pose_dim = poses.shape
        device = poses.device

        poses_reshaped = poses.reshape(B, L, J*pose_dim)
        pose_emb = self.pose_embedding(poses_reshaped)
        if self.cpe:
            pos_enc = sinusoidal_positional_encoding(L, self.d_model, device=device)
            pose_emb = pose_emb + pos_enc
        else:
            frame_idx_enc = positional_encoding_from_idx(frame_idx, self.d_model, device=device)
            pose_emb = pose_emb + frame_idx_enc

            hand_type_enc = positional_encoding_from_idx(rlh_seg, self.d_model, device=device)
            pose_emb = pose_emb + hand_type_enc

        src_attn_mask = get_batch_attention_mask(poses.reshape(B, L, -1), constants.MINUS_TWO_VALUE)
        memory = self.encoder(pose_emb, src_key_padding_mask=src_attn_mask)

        # Initialize output tokens with BOS
        output_ids = torch.full(
            (B, max_len+1), pad_token_id,
            dtype=torch.long, device=device
        )
        output_ids[:, 0] = bos_token_id

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        cross_attn_tokens = [[[] for _ in range(B)] for _ in range(self.decoder.num_layers)]
        out_features = torch.full(
            (B, max_len, self.char_size),
            constants.MINUS_ONE_HUNDRED_VALUE,
            dtype=torch.float,
            device=device
        )
        out_pos_idx = torch.full(
            (B, max_len),
            pad_pos_idx,
            dtype=torch.long,
            device=device
        )

        for t in range(max_len):
            L_t = t + 1
            tgt_emb = self.char_embedding(output_ids[:, :L_t])
            pos_enc_tgt = sinusoidal_positional_encoding(L_t, self.d_model, device=device)
            tgt_emb = tgt_emb + pos_enc_tgt

            # Causal mask
            tgt_mask = torch.triu(torch.ones(L_t, L_t, device=device), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

            tgt_key_padding_mask = get_batch_attention_mask(output_ids[:, :L_t], self.char_size)

            # Decode
            out, cross_attns = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_attn_mask,
            )  # (B, L_t, d_model)

            logits = self.lm_head(out[:, -1, :])  # (B, vocab_size)
            next_token = logits.argmax(dim=-1)  # (B)

            for b in range(B):
                if not finished[b]:
                    output_ids[b, t+1] = next_token[b]
                    for l, attn in enumerate(cross_attns):
                        cross_attn_tokens[l][b].append(attn[b].detach().cpu())
                    out_features[b, t] = logits[b].softmax(0)
                    out_pos_idx[b, t] = t
                    if next_token[b].item() == eos_token_id:
                        finished[b] = True

            if finished.all():
                break

        output = {}
        output['output_ids'] = output_ids
        output['encoder_memory'] = memory
        output['cross_attns'] = cross_attn_tokens
        output['out_features'] = out_features
        output['out_pos_idx'] = out_pos_idx
        return output