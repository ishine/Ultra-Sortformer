"""
Sortformer Autoresearch — the single file the agent modifies.
Contains: extracted model modules, loss, hyperparameters, training loop, post-processing.
Usage: python train.py
"""

import gc
import itertools
import math
import os
import random
import sys
import time
import types
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "NeMo"))

from prepare import (
    BASE_MODEL_PATH,
    FIXED_STEPS,
    MAX_NUM_SPKS,
    SAMPLE_RATE,
    SRC_NUM_SPKS,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    evaluate_der,
    load_base_model,
)

# ===================================================================
# HYPERPARAMETERS — edit freely
# ===================================================================

LR = 1e-5
OPTIM_NEW_LR = 1e-4
BATCH_SIZE = 4
SESSION_LEN_SEC = 180
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-9

FREEZE_ENCODER = True
FREEZE_TRANSFORMER = False

PIL_WEIGHT = 0.55
ATS_WEIGHT = 0.45

BASE_SPEECH_PROB_THRESHOLD = 0.54
NEW_SPEECH_PROB_THRESHOLD = 0.44

FOCAL_GAMMA = 1.5
NEW_SPK_INIT_NOISE = 0.05

N_BASE_SPKS = 4
NUM_SPKS = 8

SEED = 42

# --- Inference post-processing (diarize / eval only; training path unchanged) ---
PP_ENABLE = True
PP_MEDIAN_KERNEL = 9
PP_MORPH_KERNEL = 5
PP_MORPH_BIN_THRESH = 0.44
PP_MORPH_FILL_PROB = 0.56
PP_GAP_MAX_FRAMES = 6
PP_GAP_BIN_THRESH = 0.40
PP_GAP_BRIDGE_PROB = 0.50
PP_AVG_SMOOTH_KERNEL = 11
USE_ALIBI_REL_BIAS = False
USE_ROPE = True
ROPE_THETA = 10000.0

DECORR_WEIGHT = 0.005

# ===================================================================
# Extracted modules — agent can modify these
# ===================================================================

# -------------------------------------------------------------------
# form_attention_mask (from NeMo common parts)
# -------------------------------------------------------------------

NEG_INF = -10000.0


def form_attention_mask(input_mask, diagonal=None):
    """Build attention mask with optional future-token masking."""
    if input_mask is None:
        return None
    attn_shape = (1, input_mask.shape[1], input_mask.shape[1])
    attn_mask = input_mask.to(dtype=bool).unsqueeze(1)
    if diagonal is not None:
        future_mask = torch.tril(
            torch.ones(attn_shape, dtype=torch.bool, device=input_mask.device), diagonal
        )
        attn_mask = attn_mask & future_mask
    attention_mask = (1 - attn_mask.to(torch.float)) * NEG_INF
    return attention_mask.unsqueeze(1)


def _apply_rope_qk(
    q: torch.Tensor, k: torch.Tensor, theta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE on last dim of q,k; shapes [B, H, T, D]."""
    _, _, t, d = q.shape
    if d % 2 != 0:
        return q, k
    device = q.device
    dtype = q.dtype
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / float(d)))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(dtype).view(1, 1, t, d // 2)
    sin = freqs.sin().to(dtype).view(1, 1, t, d // 2)
    q1, q2 = q[..., : d // 2], q[..., d // 2 :]
    k1, k2 = k[..., : d // 2], k[..., d // 2 :]
    q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_out, k_out


# -------------------------------------------------------------------
# MultiHeadAttention (from NeMo transformer_modules)
# -------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attn_score_dropout=0.0, attn_layer_dropout=0.0,
                 alibi_bias=False, rope=False, rope_theta=10000.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) not divisible by num_attention_heads ({num_attention_heads})"
            )
        if alibi_bias and rope:
            raise ValueError("alibi_bias and rope cannot both be True")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = hidden_size // num_attention_heads
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))
        self.alibi_bias = alibi_bias
        self.rope = rope
        self.rope_theta = rope_theta
        if alibi_bias:
            self.alibi_slope = nn.Parameter(torch.full((num_attention_heads,), 0.08))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):
        query = self.transpose_for_scores(self.query_net(queries)) / self.attn_scale
        key = self.transpose_for_scores(self.key_net(keys)) / self.attn_scale
        value = self.transpose_for_scores(self.value_net(values))

        if self.rope:
            query, key = _apply_rope_qk(query, key, self.rope_theta)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if self.alibi_bias:
            tlen = attention_scores.size(-1)
            idx = torch.arange(tlen, device=attention_scores.device, dtype=attention_scores.dtype)
            dist = (idx.view(1, -1) - idx.view(-1, 1)).abs()
            rel = -self.alibi_slope.view(-1, 1, 1) * dist.unsqueeze(0)
            attention_scores = attention_scores + rel.unsqueeze(0)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_shape)

        output = self.out_projection(context)
        output = self.layer_dropout(output)
        return output, {}


# -------------------------------------------------------------------
# PositionWiseFF (from NeMo transformer_modules)
# -------------------------------------------------------------------

class PositionWiseFF(nn.Module):
    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        act_map = {"gelu": F.gelu, "relu": torch.relu}
        self.act_fn = act_map[hidden_act]

    def forward(self, hidden_states):
        output = self.dense_in(hidden_states)
        output = self.act_fn(output)
        output = self.dense_out(output)
        output = self.layer_dropout(output)
        return output


# -------------------------------------------------------------------
# TransformerEncoder (from NeMo transformer_encoders)
# -------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, inner_size, num_attention_heads=1,
                 attn_score_dropout=0.0, attn_layer_dropout=0.0,
                 ffn_dropout=0.0, hidden_act="relu", pre_ln=False,
                 alibi_bias=False, rope=False, rope_theta=10000.0):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout,
            alibi_bias=alibi_bias, rope=rope, rope_theta=rope_theta,
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)
        self.ls1 = nn.Parameter(torch.tensor(1.0))
        self.ls2 = nn.Parameter(torch.tensor(1.0))

    def forward_preln(self, encoder_query, encoder_mask, encoder_keys):
        residual = encoder_query
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output, _ = self.first_sub_layer(
            encoder_query, encoder_keys, encoder_keys, encoder_mask
        )
        self_attn_output += residual
        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output = self.second_sub_layer(self_attn_output)
        output += residual
        return output

    def forward_postln(self, encoder_query, encoder_mask, encoder_keys):
        attn_out, _ = self.first_sub_layer(
            encoder_query, encoder_keys, encoder_keys, encoder_mask
        )
        self_attn_output = encoder_query + self.ls1 * attn_out
        self_attn_output = self.layer_norm_1(self_attn_output)
        ffn_out = self.second_sub_layer(self_attn_output)
        output = self_attn_output + self.ls2 * ffn_out
        output = self.layer_norm_2(output)
        return output

    def forward(self, encoder_query, encoder_mask, encoder_keys):
        if self.pre_ln:
            return self.forward_preln(encoder_query, encoder_mask, encoder_keys)
        return self.forward_postln(encoder_query, encoder_mask, encoder_keys)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, inner_size, mask_future=False,
                 num_attention_heads=1, attn_score_dropout=0.0,
                 attn_layer_dropout=0.0, ffn_dropout=0.0,
                 hidden_act="relu", pre_ln=False, pre_ln_final_layer_norm=True,
                 alibi_bias=False, rope=False, rope_theta=10000.0):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        self.d_model = hidden_size
        layer = TransformerEncoderBlock(
            hidden_size, inner_size, num_attention_heads,
            attn_score_dropout, attn_layer_dropout,
            ffn_dropout, hidden_act, pre_ln,
            alibi_bias=alibi_bias, rope=rope, rope_theta=rope_theta,
        )
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)
        memory_states = encoder_states
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[0], encoder_states), dim=1)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
            if encoder_mems_list is not None:
                memory_states = torch.cat((encoder_mems_list[i + 1], encoder_states), dim=1)
            else:
                memory_states = encoder_states
            cached_mems_list.append(memory_states)

        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)
            if encoder_mems_list is not None:
                memory_states = torch.cat((encoder_mems_list[i + 1], encoder_states), dim=1)
            else:
                memory_states = encoder_states
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        return cached_mems_list[-1]


# -------------------------------------------------------------------
# SortformerModules (from NeMo sortformer_modules — core innovation area)
# -------------------------------------------------------------------

@dataclass
class StreamingSortformerState:
    spkcache: Optional[torch.Tensor] = None
    spkcache_lengths: Optional[torch.Tensor] = None
    spkcache_preds: Optional[torch.Tensor] = None
    fifo: Optional[torch.Tensor] = None
    fifo_lengths: Optional[torch.Tensor] = None
    fifo_preds: Optional[torch.Tensor] = None
    spk_perm: Optional[torch.Tensor] = None
    mean_sil_emb: Optional[torch.Tensor] = None
    n_sil_frames: Optional[torch.Tensor] = None


class SortformerModules(nn.Module):
    """Extracted from NeMo SortformerModules — agent can modify freely."""

    def __init__(
        self,
        num_spks=8,
        dropout_rate=0.5,
        fc_d_model=512,
        tf_d_model=192,
        subsampling_factor=8,
        spkcache_len=376,
        fifo_len=0,
        chunk_len=376,
        spkcache_update_period=376,
        chunk_left_context=1,
        chunk_right_context=1,
        spkcache_sil_frames_per_spk=3,
        scores_add_rnd=0.0,
        pred_score_threshold=0.25,
        max_index=99999,
        scores_boost_latest=0.05,
        sil_threshold=0.2,
        strong_boost_rate=0.75,
        weak_boost_rate=1.5,
        min_pos_scores_rate=0.5,
        n_base_spks=0,
        causal_attn_rate=0.0,
        causal_attn_rc=7,
        base_speech_prob_threshold=0.5,
        new_speech_prob_threshold=0.5,
        log=False,
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.n_spk = num_spks
        self.n_base_spks = n_base_spks

        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.n_spk)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)

        if self.n_base_spks > 0:
            n_new = self.n_spk - self.n_base_spks
            self.single_hidden_to_spks_base = nn.Linear(self.hidden_size, self.n_base_spks)
            self.single_hidden_to_spks_new = nn.Linear(self.hidden_size, n_new)
        else:
            self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.n_spk)

        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)

        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.chunk_len = chunk_len
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context
        self.spkcache_sil_frames_per_spk = spkcache_sil_frames_per_spk
        self.spkcache_update_period = spkcache_update_period
        self.causal_attn_rate = causal_attn_rate
        self.causal_attn_rc = causal_attn_rc
        self.scores_add_rnd = scores_add_rnd
        self.max_index = max_index
        self.pred_score_threshold = pred_score_threshold
        self.scores_boost_latest = scores_boost_latest
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.base_speech_prob_threshold = base_speech_prob_threshold
        self.new_speech_prob_threshold = new_speech_prob_threshold
        self.log = log

    def _check_streaming_parameters(self):
        param_constraints = {
            'spkcache_len': (1 + self.spkcache_sil_frames_per_spk) * self.n_spk,
            'fifo_len': 0,
            'chunk_len': 1,
            'spkcache_update_period': 1,
            'chunk_left_context': 0,
            'chunk_right_context': 0,
            'spkcache_sil_frames_per_spk': 0,
        }
        for param, min_val in param_constraints.items():
            val = getattr(self, param)
            if not isinstance(val, int):
                raise TypeError(f"Parameter '{param}' must be an integer, but got {param}: {val}")
            if val < min_val:
                raise ValueError(f"Parameter '{param}' must be at least {min_val}, but got {val}.")

    @staticmethod
    def length_to_mask(lengths, max_length):
        batch_size = lengths.shape[0]
        arange = torch.arange(max_length, device=lengths.device)
        mask = arange.expand(batch_size, max_length) < lengths.unsqueeze(1)
        return mask

    @staticmethod
    def concat_and_pad(embs: List[torch.Tensor], lengths: List[torch.Tensor]):
        if len(embs) != len(lengths):
            raise ValueError(
                "Length lists must have the same length, but got "
                + str(len(embs)) + " and " + str(len(lengths)) + "."
            )
        if len(embs) == 0:
            raise ValueError("Cannot concatenate empty lists.")
        device = embs[0].device
        dtype = embs[0].dtype
        batch_size = embs[0].shape[0]
        emb_dim = embs[0].shape[2]
        total_lengths = torch.sum(torch.stack(lengths), dim=0)
        sig_length = total_lengths.max().item()
        output = torch.zeros(batch_size, sig_length, emb_dim, device=device, dtype=dtype)
        start_indices = torch.zeros(batch_size, dtype=torch.int64, device=device)
        for i in range(len(embs)):
            emb = embs[i]
            length = lengths[i]
            end_indices = start_indices + length
            for batch_idx in range(batch_size):
                output[batch_idx, start_indices[batch_idx]:end_indices[batch_idx]] = emb[batch_idx, :length[batch_idx]]
            start_indices = end_indices
        return output, total_lengths

    def forward_speaker_sigmoids(self, hidden_out):
        """Final layer: sigmoid speaker probabilities. Agent can modify this."""
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        if self.n_base_spks > 0:
            spk_preds = torch.cat(
                [self.single_hidden_to_spks_base(hidden_out), self.single_hidden_to_spks_new(hidden_out)],
                dim=-1,
            )
        else:
            spk_preds = self.single_hidden_to_spks(hidden_out)
        return torch.sigmoid(spk_preds)

    @staticmethod
    def concat_embs(list_of_tensors, return_lengths=False, dim=1, device=None):
        embs = torch.cat(list_of_tensors, dim=dim).to(device)
        lengths = torch.tensor(embs.shape[1]).repeat(embs.shape[0]).to(device)
        if return_lengths:
            return embs, lengths
        return embs

    def streaming_feat_loader(self, feat_seq, feat_seq_length, feat_seq_offset):
        feat_len = feat_seq.shape[2]
        stt_feat, end_feat, chunk_idx = 0, 0, 0
        while end_feat < feat_len:
            left_offset = min(self.chunk_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + self.chunk_len * self.subsampling_factor, feat_len)
            right_offset = min(self.chunk_right_context * self.subsampling_factor, feat_len - end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat - left_offset : end_feat + right_offset]
            feat_lengths = (feat_seq_length + feat_seq_offset - stt_feat + left_offset).clamp(
                0, chunk_feat_seq.shape[2]
            )
            feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
            stt_feat = end_feat
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            yield chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
            chunk_idx += 1

    def init_streaming_state(self, batch_size=1, async_streaming=False, device=None):
        state = StreamingSortformerState()
        if async_streaming:
            state.spkcache = torch.zeros((batch_size, self.spkcache_len, self.fc_d_model), device=device)
            state.spkcache_preds = torch.zeros((batch_size, self.spkcache_len, self.n_spk), device=device)
            state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
            state.fifo = torch.zeros((batch_size, self.fifo_len, self.fc_d_model), device=device)
            state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        else:
            state.spkcache = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
            state.fifo = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
        state.mean_sil_emb = torch.zeros((batch_size, self.fc_d_model), device=device)
        state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)
        return state

    @staticmethod
    def apply_mask_to_preds(preds, lengths):
        batch_size, n_frames, n_spk = preds.shape
        mask = torch.arange(n_frames, device=preds.device).view(1, -1, 1)
        mask = mask.expand(batch_size, -1, n_spk) < lengths.view(-1, 1, 1)
        return torch.where(mask, preds, torch.tensor(0.0, device=preds.device))

    def streaming_update(self, streaming_state, chunk, preds, lc=0, rc=0):
        """Synchronous streaming update. Agent can modify this logic."""
        batch_size, _, emb_dim = chunk.shape
        spkcache_len = streaming_state.spkcache.shape[1]
        fifo_len = streaming_state.fifo.shape[1]
        chunk_len = chunk.shape[1] - lc - rc

        if streaming_state.spk_perm is not None:
            inv_spk_perm = torch.stack(
                [torch.argsort(streaming_state.spk_perm[b]) for b in range(batch_size)]
            )
            preds = torch.stack(
                [preds[b, :, inv_spk_perm[b]] for b in range(batch_size)]
            )

        streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

        streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
        streaming_state.fifo_preds = torch.cat([streaming_state.fifo_preds, chunk_preds], dim=1)

        if fifo_len + chunk_len > self.fifo_len:
            pop_out_len = self.spkcache_update_period
            pop_out_len = max(pop_out_len, chunk_len - self.fifo_len + fifo_len)
            pop_out_len = min(pop_out_len, fifo_len + chunk_len)

            pop_out_embs = streaming_state.fifo[:, :pop_out_len]
            pop_out_preds = streaming_state.fifo_preds[:, :pop_out_len]
            streaming_state.mean_sil_emb, streaming_state.n_sil_frames = self._get_silence_profile(
                streaming_state.mean_sil_emb, streaming_state.n_sil_frames,
                pop_out_embs, pop_out_preds,
            )
            streaming_state.fifo = streaming_state.fifo[:, pop_out_len:]
            streaming_state.fifo_preds = streaming_state.fifo_preds[:, pop_out_len:]

            streaming_state.spkcache = torch.cat([streaming_state.spkcache, pop_out_embs], dim=1)
            if streaming_state.spkcache_preds is not None:
                streaming_state.spkcache_preds = torch.cat(
                    [streaming_state.spkcache_preds, pop_out_preds], dim=1
                )
            if streaming_state.spkcache.shape[1] > self.spkcache_len:
                if streaming_state.spkcache_preds is None:
                    streaming_state.spkcache_preds = torch.cat(
                        [preds[:, :spkcache_len], pop_out_preds], dim=1
                    )
                streaming_state.spkcache, streaming_state.spkcache_preds, streaming_state.spk_perm = (
                    self._compress_spkcache(
                        emb_seq=streaming_state.spkcache,
                        preds=streaming_state.spkcache_preds,
                        mean_sil_emb=streaming_state.mean_sil_emb,
                        permute_spk=self.training,
                    )
                )

        return streaming_state, chunk_preds

    def streaming_update_async(self, streaming_state, chunk, chunk_lengths, preds, lc=0, rc=0):
        """Asynchronous streaming update for variable-length batches."""
        batch_size, _, emb_dim = chunk.shape
        n_spk = preds.shape[2]
        max_spkcache_len = streaming_state.spkcache.shape[1]
        max_fifo_len = streaming_state.fifo.shape[1]
        max_chunk_len = chunk.shape[1] - lc - rc

        max_pop_out_len = max(self.spkcache_update_period, max_chunk_len)
        max_pop_out_len = min(max_pop_out_len, max_chunk_len + max_fifo_len)

        streaming_state.fifo_preds = torch.zeros((batch_size, max_fifo_len, n_spk), device=preds.device)
        chunk_preds = torch.zeros((batch_size, max_chunk_len, n_spk), device=preds.device)
        chunk_lengths = (chunk_lengths - lc).clamp(min=0, max=max_chunk_len)
        updated_fifo = torch.zeros((batch_size, max_fifo_len + max_chunk_len, emb_dim), device=preds.device)
        updated_fifo_preds = torch.zeros((batch_size, max_fifo_len + max_chunk_len, n_spk), device=preds.device)
        updated_spkcache = torch.zeros((batch_size, max_spkcache_len + max_pop_out_len, emb_dim), device=preds.device)
        updated_spkcache_preds = torch.full(
            (batch_size, max_spkcache_len + max_pop_out_len, n_spk), 0.0, device=preds.device
        )

        for bi in range(batch_size):
            sc_len = streaming_state.spkcache_lengths[bi].item()
            fi_len = streaming_state.fifo_lengths[bi].item()
            ch_len = chunk_lengths[bi].item()

            streaming_state.fifo_preds[bi, :fi_len, :] = preds[bi, sc_len:sc_len + fi_len, :]
            chunk_preds[bi, :ch_len, :] = preds[bi, sc_len + fi_len + lc:sc_len + fi_len + lc + ch_len]
            updated_spkcache[bi, :sc_len, :] = streaming_state.spkcache[bi, :sc_len, :]
            updated_spkcache_preds[bi, :sc_len, :] = streaming_state.spkcache_preds[bi, :sc_len, :]
            updated_fifo[bi, :fi_len, :] = streaming_state.fifo[bi, :fi_len, :]
            updated_fifo_preds[bi, :fi_len, :] = streaming_state.fifo_preds[bi, :fi_len, :]

            streaming_state.fifo_lengths[bi] += ch_len
            updated_fifo[bi, fi_len:fi_len + ch_len, :] = chunk[bi, lc:lc + ch_len, :]
            updated_fifo_preds[bi, fi_len:fi_len + ch_len, :] = chunk_preds[bi, :ch_len, :]

            if fi_len + ch_len > max_fifo_len:
                pop_out_len = self.spkcache_update_period
                pop_out_len = max(pop_out_len, max_chunk_len - max_fifo_len + fi_len)
                pop_out_len = min(pop_out_len, fi_len + ch_len)
                streaming_state.spkcache_lengths[bi] += pop_out_len
                pop_out_embs = updated_fifo[bi, :pop_out_len, :]
                pop_out_preds = updated_fifo_preds[bi, :pop_out_len, :]
                (
                    streaming_state.mean_sil_emb[bi:bi + 1],
                    streaming_state.n_sil_frames[bi:bi + 1],
                ) = self._get_silence_profile(
                    streaming_state.mean_sil_emb[bi:bi + 1],
                    streaming_state.n_sil_frames[bi:bi + 1],
                    pop_out_embs.unsqueeze(0),
                    pop_out_preds.unsqueeze(0),
                )
                updated_spkcache[bi, sc_len:sc_len + pop_out_len, :] = pop_out_embs
                if updated_spkcache_preds[bi, 0, 0] >= 0:
                    updated_spkcache_preds[bi, sc_len:sc_len + pop_out_len, :] = pop_out_preds
                elif sc_len + pop_out_len > self.spkcache_len:
                    updated_spkcache_preds[bi, :sc_len, :] = preds[bi, :sc_len, :]
                    updated_spkcache_preds[bi, sc_len:sc_len + pop_out_len, :] = pop_out_preds
                streaming_state.fifo_lengths[bi] -= pop_out_len
                new_fi_len = streaming_state.fifo_lengths[bi].item()
                updated_fifo[bi, :new_fi_len, :] = updated_fifo[bi, pop_out_len:pop_out_len + new_fi_len, :].clone()
                updated_fifo_preds[bi, :new_fi_len, :] = updated_fifo_preds[bi, pop_out_len:pop_out_len + new_fi_len, :].clone()
                updated_fifo[bi, new_fi_len:, :] = 0
                updated_fifo_preds[bi, new_fi_len:, :] = 0

        streaming_state.fifo = updated_fifo[:, :max_fifo_len, :]
        streaming_state.fifo_preds = updated_fifo_preds[:, :max_fifo_len, :]

        need_compress = streaming_state.spkcache_lengths > self.spkcache_len
        streaming_state.spkcache = updated_spkcache[:, :self.spkcache_len, :]
        streaming_state.spkcache_preds = updated_spkcache_preds[:, :self.spkcache_len, :]

        idx = torch.where(need_compress)[0]
        if len(idx) > 0:
            streaming_state.spkcache[idx], streaming_state.spkcache_preds[idx], _ = self._compress_spkcache(
                emb_seq=updated_spkcache[idx],
                preds=updated_spkcache_preds[idx],
                mean_sil_emb=streaming_state.mean_sil_emb[idx],
                permute_spk=False,
            )
            streaming_state.spkcache_lengths[idx] = streaming_state.spkcache_lengths[idx].clamp(max=self.spkcache_len)

        return streaming_state, chunk_preds

    def _get_silence_profile(self, mean_sil_emb, n_sil_frames, emb_seq, preds):
        is_sil = preds.sum(dim=2) < self.sil_threshold
        sil_count = is_sil.sum(dim=1)
        has_new_sil = sil_count > 0
        if not has_new_sil.any():
            return mean_sil_emb, n_sil_frames
        sil_emb_sum = torch.sum(emb_seq * is_sil.unsqueeze(-1), dim=1)
        upd_n = n_sil_frames + sil_count
        old_sum = mean_sil_emb * n_sil_frames.unsqueeze(1)
        total = old_sum + sil_emb_sum
        upd_mean = total / torch.clamp(upd_n.unsqueeze(1), min=1)
        return upd_mean, upd_n

    def _get_log_pred_scores(self, preds):
        """Log-based scoring for speaker cache compression. Agent can modify."""
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        scores = log_probs - log_1_probs + log_1_sum - math.log(0.5)
        return scores

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk):
        n_spk = preds.shape[2]
        # base/new 스피커마다 서로 다른 임계값 적용.
        if self.n_base_spks > 0:
            thresh = torch.full(
                (n_spk,),
                float(self.new_speech_prob_threshold),
                dtype=preds.dtype,
                device=preds.device,
            )
            thresh[: self.n_base_spks] = float(self.base_speech_prob_threshold)
        else:
            thresh = torch.full(
                (n_spk,),
                float(self.new_speech_prob_threshold),
                dtype=preds.dtype,
                device=preds.device,
            )
        is_speech = preds > thresh.view(1, 1, -1)
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf'), device=scores.device))
        is_pos = scores > 0
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float('-inf'), device=scores.device), scores)
        return scores

    def _boost_topk_scores(self, scores, n_boost_per_spk, scale_factor=1.0, offset=0.5):
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        batch_idx = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        spk_idx = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)
        scores[batch_idx, topk_indices, spk_idx] -= scale_factor * math.log(offset)
        return scores

    def _get_topk_indices(self, scores):
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self.spkcache_sil_frames_per_spk
        scores_flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flat, self.spkcache_len, dim=1, sorted=False)
        valid = topk_values != float('-inf')
        topk_indices = torch.where(valid, topk_indices, torch.tensor(self.max_index, device=scores.device))
        topk_sorted, _ = torch.sort(topk_indices, dim=1)
        is_disabled = topk_sorted == self.max_index
        topk_sorted = torch.remainder(topk_sorted, n_frames)
        is_disabled = is_disabled | (topk_sorted >= n_frames_no_sil)
        topk_sorted[is_disabled] = 0
        return topk_sorted, is_disabled

    def _gather_spkcache_and_preds(self, emb_seq, preds, topk_indices, is_disabled, mean_sil_emb):
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        idx_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        gathered_emb = torch.gather(emb_seq, 1, idx_emb)
        sil_expanded = mean_sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        gathered_emb = torch.where(is_disabled.unsqueeze(-1), sil_expanded, gathered_emb)

        idx_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        gathered_preds = torch.gather(preds, 1, idx_spk)
        gathered_preds = torch.where(
            is_disabled.unsqueeze(-1), torch.tensor(0.0, device=preds.device), gathered_preds
        )
        return gathered_emb, gathered_preds

    def _get_max_perm_index(self, scores):
        batch_size, _, n_spk = scores.shape
        is_pos = scores > 0
        zero_indices = torch.where(is_pos.sum(dim=1) == 0)
        max_perm = torch.full((batch_size,), n_spk, dtype=torch.long, device=scores.device)
        max_perm.scatter_reduce_(0, zero_indices[0], zero_indices[1], reduce="amin", include_self=False)
        return max_perm

    def _permute_speakers(self, scores, max_perm_index):
        spk_perm_list, scores_list = [], []
        batch_size, _, n_spk = scores.shape
        for b in range(batch_size):
            rand_perm = torch.randperm(max_perm_index[b].item())
            linear = torch.arange(max_perm_index[b].item(), n_spk)
            perm = torch.cat([rand_perm, linear])
            spk_perm_list.append(perm)
            scores_list.append(scores[b, :, perm])
        spk_perm = torch.stack(spk_perm_list).to(scores.device)
        scores = torch.stack(scores_list).to(scores.device)
        return scores, spk_perm

    def _compress_spkcache(self, emb_seq, preds, mean_sil_emb, permute_spk=False):
        """Speaker cache compression. Agent can modify this algorithm."""
        batch_size, n_frames, n_spk = preds.shape
        spkcache_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong_boost = math.floor(spkcache_per_spk * self.strong_boost_rate)
        weak_boost = math.floor(spkcache_per_spk * self.weak_boost_rate)
        min_pos = math.floor(spkcache_per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos)

        if permute_spk:
            max_perm = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm)
        else:
            spk_perm = None

        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len:, :] += self.scores_boost_latest

        if self.training and self.scores_add_rnd > 0:
            scores += torch.rand_like(scores) * self.scores_add_rnd

        scores = self._boost_topk_scores(scores, strong_boost, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak_boost, scale_factor=1)

        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full(
                (batch_size, self.spkcache_sil_frames_per_spk, n_spk),
                float('inf'), device=scores.device,
            )
            scores = torch.cat([scores, pad], dim=1)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )
        return spkcache, spkcache_preds, spk_perm


# -------------------------------------------------------------------
# BCELoss (from NeMo bce_loss — agent can replace with novel losses)
# -------------------------------------------------------------------

class BCELoss(nn.Module):
    """Binary cross entropy loss for speaker diarization."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_f = nn.BCELoss(reduction=reduction)

    def forward(self, probs, labels, target_lens):
        probs_list = [probs[k, :target_lens[k], :] for k in range(probs.shape[0])]
        labels_list = [labels[k, :target_lens[k], :] for k in range(labels.shape[0])]
        probs_cat = torch.cat(probs_list, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)
        return self.loss_f(probs_cat, labels_cat)


class FocalBCELoss(nn.Module):
    """Focal loss on frame-wise BCE — down-weights easy frames, focuses on hard negatives/positives."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, probs, labels, target_lens):
        eps = 1e-7
        probs_list = [probs[k, :target_lens[k], :] for k in range(probs.shape[0])]
        labels_list = [labels[k, :target_lens[k], :] for k in range(labels.shape[0])]
        p = torch.cat(probs_list, dim=0)
        y = torch.cat(labels_list, dim=0)
        bce = -(y * torch.log(p + eps) + (1.0 - y) * torch.log(1.0 - p + eps))
        pt = p * y + (1.0 - p) * (1.0 - y)
        pt = pt.clamp(min=eps, max=1.0 - eps)
        w = (1.0 - pt).pow(self.gamma)
        return (w * bce).mean()


# ===================================================================
# MODEL EXPANSION — agent can completely replace this section
# ===================================================================
# This section controls how the 4spk model is expanded to 8spk.
# The agent can try entirely different expansion strategies:
#   - SVD orthogonal init (current default)
#   - Random init with specific distributions
#   - Knowledge distillation from 4spk outputs
#   - Adapter-based expansion (freeze base, add small adapters)
#   - Shared projection + per-speaker heads
#   - Progressive expansion (4→5→6→...→8)
#   - Copy and perturb (clone existing weights + noise)

def init_new_speaker_weights(existing_weight, n_new_rows):
    """Initialize weight rows for new speakers 5-8.
    Agent can replace this with any initialization strategy.

    Args:
        existing_weight: (src_spks, hidden_dim) weight matrix from 4spk model
        n_new_rows: number of new speaker rows to create (dst_spks - src_spks)
    Returns:
        (n_new_rows, hidden_dim) tensor for new speaker weights
    """
    existing = existing_weight.float()
    avg_norm = torch.norm(existing, p=2, dim=1).mean()
    avg_mean = existing.mean()
    avg_std = existing.std().clamp(min=1e-6)

    _, _, Vh = torch.linalg.svd(existing, full_matrices=True)

    new_rows = []
    n_src = existing.shape[0]
    for i in range(n_new_rows):
        idx = n_src + i
        if idx < Vh.shape[0]:
            new_vec = Vh[idx : idx + 1, :].clone()
        else:
            new_vec = existing.mean(dim=0, keepdim=True) + torch.randn_like(existing[:1]) * avg_std
        new_vec = (new_vec - new_vec.mean()) / (new_vec.std() + 1e-6)
        new_vec = new_vec * avg_std + avg_mean
        current_norm = torch.norm(new_vec, p=2)
        if current_norm > 1e-6:
            new_vec = new_vec * (avg_norm / current_norm)
        new_rows.append(new_vec)
    out = torch.cat(new_rows, dim=0).to(existing_weight.dtype)
    noise = torch.randn_like(out, dtype=torch.float32) * (avg_std * NEW_SPK_INIT_NOISE)
    out = (out.float() + noise).to(out.dtype)
    return out


def init_new_speaker_bias(existing_bias, n_new):
    """Initialize bias values for new speakers 5-8.
    Agent can replace this with any initialization strategy.
    """
    new_bias = []
    for _ in range(n_new):
        val = existing_bias.mean() + torch.randn(1, device=existing_bias.device, dtype=existing_bias.dtype).squeeze() * existing_bias.std()
        new_bias.append(val.unsqueeze(0))
    return torch.cat(new_bias)


def expand_sortformer_4to8(src_state, custom_sm, src_spks, dst_spks):
    """Expand a 4spk SortformerModules state_dict into an 8spk custom module.

    This is the main expansion function. Agent can completely rewrite this
    to try different expansion strategies — different architectures, different
    initialization, or entirely different approaches.

    Args:
        src_state: state_dict from the original 4spk NeMo SortformerModules
        custom_sm: the new 8spk SortformerModules instance (freshly initialized)
        src_spks: source speaker count (4)
        dst_spks: target speaker count (8)
    Returns:
        custom_sm with weights loaded/initialized
    """
    n_new = dst_spks - src_spks

    with torch.no_grad():
        custom_sm.encoder_proj.load_state_dict({
            "weight": src_state["encoder_proj.weight"],
            "bias": src_state["encoder_proj.bias"],
        })
        custom_sm.first_hidden_to_hidden.load_state_dict({
            "weight": src_state["first_hidden_to_hidden.weight"],
            "bias": src_state["first_hidden_to_hidden.bias"],
        })

        src_h2s_w = src_state["hidden_to_spks.weight"]
        src_h2s_b = src_state["hidden_to_spks.bias"]
        expanded_h2s_w = torch.cat([src_h2s_w, init_new_speaker_weights(src_h2s_w, n_new)], dim=0)
        expanded_h2s_b = torch.cat([src_h2s_b, init_new_speaker_bias(src_h2s_b, n_new)])
        custom_sm.hidden_to_spks.weight.copy_(expanded_h2s_w)
        custom_sm.hidden_to_spks.bias.copy_(expanded_h2s_b)

        src_single_w = src_state["single_hidden_to_spks.weight"]
        src_single_b = src_state["single_hidden_to_spks.bias"]
        custom_sm.single_hidden_to_spks_base.weight.copy_(src_single_w)
        custom_sm.single_hidden_to_spks_base.bias.copy_(src_single_b)
        custom_sm.single_hidden_to_spks_new.weight.copy_(
            init_new_speaker_weights(src_single_w, n_new)
        )
        custom_sm.single_hidden_to_spks_new.bias.copy_(
            init_new_speaker_bias(src_single_b, n_new)
        )

    print(f"  SortformerModules: {src_spks}spk → {dst_spks}spk expanded")
    return custom_sm


# ===================================================================
# Speaker head decorrelation (low-λ CER regularizer)
# ===================================================================

def speaker_output_decorrelation_loss(sm: SortformerModules) -> torch.Tensor:
    if sm.n_base_spks <= 0:
        dev = next(sm.parameters()).device
        dt = next(sm.parameters()).dtype
        return torch.zeros((), device=dev, dtype=dt)
    w = torch.cat([sm.single_hidden_to_spks_base.weight, sm.single_hidden_to_spks_new.weight], dim=0)
    wn = F.normalize(w, dim=1)
    g = torch.mm(wn, wn.t())
    n = g.size(0)
    mask = 1.0 - torch.eye(n, device=g.device, dtype=g.dtype)
    return (g * mask).pow(2).sum() / (n * (n - 1))


# ===================================================================
# Model assembly
# ===================================================================

def build_custom_modules(nemo_model, src_spks=SRC_NUM_SPKS, dst_spks=NUM_SPKS):
    """Build custom modules with 4spk→8spk expansion.

    Creates fresh modules, loads pretrained weights, and expands speaker outputs.
    """
    cfg = nemo_model._cfg
    sm_cfg = cfg.sortformer_modules

    custom_sm = SortformerModules(
        num_spks=dst_spks,
        dropout_rate=sm_cfg.get("dropout_rate", 0.5),
        fc_d_model=sm_cfg.get("fc_d_model", 512),
        tf_d_model=sm_cfg.get("tf_d_model", 192),
        subsampling_factor=sm_cfg.get("subsampling_factor", 8),
        spkcache_len=sm_cfg.get("spkcache_len", 376),
        fifo_len=sm_cfg.get("fifo_len", 0),
        chunk_len=sm_cfg.get("chunk_len", 376),
        spkcache_update_period=sm_cfg.get("spkcache_update_period", 376),
        chunk_left_context=sm_cfg.get("chunk_left_context", 1),
        chunk_right_context=sm_cfg.get("chunk_right_context", 1),
        spkcache_sil_frames_per_spk=sm_cfg.get("spkcache_sil_frames_per_spk", 3),
        scores_add_rnd=sm_cfg.get("scores_add_rnd", 0.0),
        pred_score_threshold=sm_cfg.get("pred_score_threshold", 0.25),
        max_index=sm_cfg.get("max_index", 99999),
        scores_boost_latest=sm_cfg.get("scores_boost_latest", 0.05),
        sil_threshold=sm_cfg.get("sil_threshold", 0.2),
        strong_boost_rate=sm_cfg.get("strong_boost_rate", 0.75),
        weak_boost_rate=sm_cfg.get("weak_boost_rate", 1.5),
        min_pos_scores_rate=sm_cfg.get("min_pos_scores_rate", 0.5),
        n_base_spks=N_BASE_SPKS,
        base_speech_prob_threshold=BASE_SPEECH_PROB_THRESHOLD,
        new_speech_prob_threshold=NEW_SPEECH_PROB_THRESHOLD,
    )

    src_state = nemo_model.sortformer_modules.state_dict()
    custom_sm = expand_sortformer_4to8(src_state, custom_sm, src_spks, dst_spks)

    tf_cfg = cfg.transformer_encoder
    custom_tf = TransformerEncoder(
        num_layers=tf_cfg.get("num_layers", 18),
        hidden_size=tf_cfg.get("hidden_size", 192),
        inner_size=tf_cfg.get("inner_size", 768),
        mask_future=tf_cfg.get("mask_future", False),
        num_attention_heads=tf_cfg.get("num_attention_heads", 8),
        attn_score_dropout=tf_cfg.get("attn_score_dropout", 0.0),
        attn_layer_dropout=tf_cfg.get("attn_layer_dropout", 0.0),
        ffn_dropout=tf_cfg.get("ffn_dropout", 0.0),
        hidden_act="gelu",
        pre_ln=tf_cfg.get("pre_ln", False),
        pre_ln_final_layer_norm=tf_cfg.get("pre_ln_final_layer_norm", True),
        alibi_bias=USE_ALIBI_REL_BIAS,
        rope=USE_ROPE,
        rope_theta=ROPE_THETA,
    )

    tf_state = nemo_model.transformer_encoder.state_dict()
    missing, unexpected = custom_tf.load_state_dict(tf_state, strict=False)
    if missing:
        print(f"  TransformerEncoder missing keys: {missing}")
    if unexpected:
        print(f"  TransformerEncoder unexpected keys: {unexpected}")
    if USE_ROPE:
        print(f"  TransformerEncoder: RoPE (theta={ROPE_THETA}), ALiBi off")

    custom_loss = FocalBCELoss(gamma=FOCAL_GAMMA)

    return custom_sm, custom_tf, custom_loss


def _approx_frame_lengths_from_audio(
    model, audio_signal_length: torch.Tensor, max_T: int, device: torch.device
) -> torch.Tensor:
    hop = max(1, int(float(model._cfg.preprocessor.window_stride) * float(model._cfg.sample_rate)))
    sub = max(1, int(model.encoder.subsampling_factor))
    al = audio_signal_length.long().to(device)
    return (al // (hop * sub)).clamp(min=1, max=max_T)


def temporal_median_filter_probs(preds: torch.Tensor, kernel: int, frame_lengths: torch.Tensor) -> torch.Tensor:
    if kernel <= 1 or (kernel % 2) == 0:
        return preds
    b, t, s = preds.shape
    pad = kernel // 2
    x = preds.permute(0, 2, 1)
    x = F.pad(x, (pad, pad), mode="reflect")
    unf = x.unfold(-1, kernel, 1)
    med = unf.median(dim=-1).values
    out = med.permute(0, 2, 1)
    ar = torch.arange(t, device=preds.device).view(1, t).expand(b, -1)
    m = (ar < frame_lengths.view(-1, 1)).unsqueeze(-1).expand(b, t, s)
    return torch.where(m, out, torch.zeros_like(preds))


def morph_close_gap_fill_probs(
    preds: torch.Tensor,
    kernel: int,
    bin_thresh: float,
    fill_prob: float,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    if kernel <= 1:
        return preds
    b, t, s = preds.shape
    pad = kernel // 2
    orig_bin = (preds >= bin_thresh).float()
    x = orig_bin.permute(0, 2, 1)
    dil = F.max_pool1d(x, kernel, stride=1, padding=pad)
    inv = 1.0 - dil
    inv_e = F.max_pool1d(inv, kernel, stride=1, padding=pad)
    closed = (1.0 - inv_e).permute(0, 2, 1)
    fill_mask = (closed > orig_bin).float()
    boosted = torch.maximum(preds, fill_mask * fill_prob)
    ar = torch.arange(t, device=preds.device).view(1, t).expand(b, -1)
    m = (ar < frame_lengths.view(-1, 1)).unsqueeze(-1).expand(b, t, s)
    return torch.where(m, boosted, torch.zeros_like(preds))


def fill_short_interior_silence_gaps(
    preds: torch.Tensor,
    frame_lengths: torch.Tensor,
    bin_thresh: float,
    max_gap: int,
    bridge_prob: float,
) -> torch.Tensor:
    """Bridge interior silence runs (≤max_gap) framed by speech on both sides → reduce MISS."""
    if max_gap < 1:
        return preds
    b, t, s = preds.shape
    out = preds.clone()
    binx = (preds >= bin_thresh).float()
    x = binx.permute(0, 2, 1).contiguous()
    fp = preds.new_tensor(bridge_prob)
    for g in range(1, max_gap + 1):
        if t <= g + 1:
            break
        w = x.unfold(-1, g + 2, 1)
        hole = (w[..., 0] > 0.5) & (w[..., -1] > 0.5) & (w[..., 1:-1].sum(dim=-1) < 0.5)
        tw = hole.shape[-1]
        for i in range(tw):
            hi = hole[:, :, i]
            if not hi.any():
                continue
            sl = slice(i + 1, i + 1 + g)
            seg = out[:, sl, :].permute(0, 2, 1)
            hb = hi.unsqueeze(-1).expand(b, s, g)
            seg = torch.where(hb, torch.maximum(seg, fp), seg)
            out[:, sl, :] = seg.permute(0, 2, 1)
    ar = torch.arange(t, device=preds.device).view(1, t).expand(b, -1)
    m = (ar < frame_lengths.view(-1, 1)).unsqueeze(-1).expand(b, t, s)
    return torch.where(m, out, torch.zeros_like(preds))


def temporal_avg_smooth_probs(preds: torch.Tensor, kernel: int, frame_lengths: torch.Tensor) -> torch.Tensor:
    if kernel <= 1 or (kernel % 2) == 0:
        return preds
    b, t, s = preds.shape
    pad = kernel // 2
    x = preds.permute(0, 2, 1)
    x = F.pad(x, (pad, pad), mode="reflect")
    sm = F.avg_pool1d(x, kernel, stride=1, padding=0)
    sm = sm.permute(0, 2, 1)
    ar = torch.arange(t, device=preds.device).view(1, t).expand(b, -1)
    m = (ar < frame_lengths.view(-1, 1)).unsqueeze(-1).expand(b, t, s)
    return torch.where(m, sm, torch.zeros_like(preds))


def apply_infer_postprocess_probs(model, preds: torch.Tensor, audio_signal_length: torch.Tensor) -> torch.Tensor:
    if not PP_ENABLE:
        return preds
    dev = preds.device
    max_t = preds.shape[1]
    fl = _approx_frame_lengths_from_audio(model, audio_signal_length, max_t, dev)
    out = temporal_median_filter_probs(preds, PP_MEDIAN_KERNEL, fl)
    out = morph_close_gap_fill_probs(
        out, PP_MORPH_KERNEL, PP_MORPH_BIN_THRESH, PP_MORPH_FILL_PROB, fl
    )
    out = fill_short_interior_silence_gaps(
        out, fl, PP_GAP_BIN_THRESH, PP_GAP_MAX_FRAMES, PP_GAP_BRIDGE_PROB
    )
    out = temporal_avg_smooth_probs(out, PP_AVG_SMOOTH_KERNEL, fl)
    return out


def patch_nemo_forward_infer_postprocess(model):
    if getattr(model, "_autoresearch_forward_orig", None) is not None:
        return
    model._autoresearch_forward_orig = model.forward

    def forward_with_pp(self, audio_signal, audio_signal_length):
        preds = self._autoresearch_forward_orig(audio_signal, audio_signal_length)
        if self.training or not PP_ENABLE:
            return preds
        with torch.no_grad():
            return apply_infer_postprocess_probs(self, preds, audio_signal_length)

    model.forward = types.MethodType(forward_with_pp, model)


def forward_infer(encoder, transformer_encoder, sortformer_modules, processed_signal, processed_signal_length):
    """Offline inference forward pass."""
    emb_seq, emb_seq_length = encoder(audio_signal=processed_signal, length=processed_signal_length)
    emb_seq = emb_seq.transpose(1, 2)
    if sortformer_modules.encoder_proj is not None:
        emb_seq = sortformer_modules.encoder_proj(emb_seq)

    encoder_mask = sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
    trans_out = transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
    preds = sortformer_modules.forward_speaker_sigmoids(trans_out)
    preds = preds * encoder_mask.unsqueeze(-1)
    return preds


def forward_streaming(encoder, transformer_encoder, sortformer_modules, processed_signal, processed_signal_length, device):
    """Streaming inference forward pass."""
    streaming_state = sortformer_modules.init_streaming_state(
        batch_size=processed_signal.shape[0], device=device
    )
    total_preds = torch.zeros((processed_signal.shape[0], 0, sortformer_modules.n_spk), device=device)
    processed_signal_offset = torch.zeros((processed_signal.shape[0],), dtype=torch.long, device=device)

    for _, chunk_feat, feat_lengths, lo, ro in sortformer_modules.streaming_feat_loader(
        processed_signal, processed_signal_length, processed_signal_offset
    ):
        chunk_pre, chunk_pre_len = encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)
        spkcache_fifo_chunk = sortformer_modules.concat_embs(
            [streaming_state.spkcache, streaming_state.fifo, chunk_pre], dim=1, device=device
        )
        total_len = streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_len

        emb, emb_len = encoder(
            audio_signal=spkcache_fifo_chunk, length=total_len, bypass_pre_encode=True
        )
        emb = emb.transpose(1, 2)
        if sortformer_modules.encoder_proj is not None:
            emb = sortformer_modules.encoder_proj(emb)

        enc_mask = sortformer_modules.length_to_mask(emb_len, emb.shape[1])
        trans_out = transformer_encoder(encoder_states=emb, encoder_mask=enc_mask)
        preds = sortformer_modules.forward_speaker_sigmoids(trans_out)
        preds = preds * enc_mask.unsqueeze(-1)
        preds = sortformer_modules.apply_mask_to_preds(preds, emb_len)

        streaming_state, chunk_preds = sortformer_modules.streaming_update(
            streaming_state, chunk_pre, preds,
            lc=round(lo / encoder.subsampling_factor),
            rc=math.ceil(ro / encoder.subsampling_factor),
        )
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

    return total_preds


# ===================================================================
# Main training script
# ===================================================================

if __name__ == "__main__":
    # --- DDP setup ---
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        is_main = True

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    t_start = time.time()

    # --- Load base model (4spk) ---
    if is_main:
        print("=" * 60)
        print(f"Loading base {SRC_NUM_SPKS}spk NeMo model...")
    nemo_model = load_base_model(BASE_MODEL_PATH, device=str(device))

    # Override config for 8-speaker operation
    from omegaconf import OmegaConf
    OmegaConf.update(nemo_model._cfg, "max_num_of_spks", MAX_NUM_SPKS)

    # --- Build custom modules (4spk → 8spk expansion) ---
    if is_main:
        print(f"Building custom modules: {SRC_NUM_SPKS}spk → {NUM_SPKS}spk expansion...")
    custom_sm, custom_tf, custom_loss = build_custom_modules(nemo_model)
    custom_sm = custom_sm.to(device)
    custom_tf = custom_tf.to(device)
    custom_loss = custom_loss.to(device)

    # Replace modules in the NeMo model
    nemo_model.sortformer_modules = custom_sm
    nemo_model.transformer_encoder = custom_tf
    nemo_model.loss = custom_loss
    nemo_model.concat_and_pad_script = torch.jit.script(custom_sm.concat_and_pad)
    patch_nemo_forward_infer_postprocess(nemo_model)

    # --- Freeze encoder if configured ---
    if FREEZE_ENCODER:
        for p in nemo_model.encoder.parameters():
            p.requires_grad = False
        for p in nemo_model.preprocessor.parameters():
            p.requires_grad = False

    if FREEZE_TRANSFORMER:
        for p in custom_tf.parameters():
            p.requires_grad = False

    # --- Count parameters ---
    trainable = sum(p.numel() for p in nemo_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in nemo_model.parameters())
    if is_main:
        print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # --- Setup optimizer ---
    param_groups = []
    if N_BASE_SPKS > 0 and hasattr(custom_sm, 'single_hidden_to_spks_new'):
        new_params = list(custom_sm.single_hidden_to_spks_new.parameters())
        new_ids = {id(p) for p in new_params}
        base_params = [p for p in nemo_model.parameters() if p.requires_grad and id(p) not in new_ids]
        param_groups = [
            {"params": base_params, "lr": LR},
            {"params": new_params, "lr": OPTIM_NEW_LR},
        ]
    else:
        param_groups = [{"params": [p for p in nemo_model.parameters() if p.requires_grad], "lr": LR}]

    optimizer = torch.optim.AdamW(param_groups, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

    # --- LR scheduler ---
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return max(0.0, 1.0 / math.sqrt(max(step, WARMUP_STEPS) / WARMUP_STEPS))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Setup data ---
    from omegaconf import DictConfig
    if is_main:
        print(f"Setting up training data: {TRAIN_MANIFEST}")
    train_data_cfg = DictConfig({
        "manifest_filepath": TRAIN_MANIFEST,
        "sample_rate": SAMPLE_RATE,
        "num_spks": MAX_NUM_SPKS,
        "session_len_sec": SESSION_LEN_SEC,
        "batch_size": BATCH_SIZE,
        "soft_label_thres": 0.5,
        "soft_targets": False,
        "labels": None,
        "shuffle": not use_ddp,
        "num_workers": 4,
        "validation_mode": False,
        "use_lhotse": False,
        "use_bucketing": False,
        "pin_memory": True,
        "drop_last": True,
        "window_stride": 0.01,
        "subsampling_factor": 8,
    })
    nemo_model.setup_training_data(train_data_config=train_data_cfg)
    train_dl = nemo_model._train_dl
    if train_dl is None:
        if is_main:
            print("FAIL: Could not create training dataloader")
        sys.exit(1)

    if use_ddp:
        train_sampler = DistributedSampler(train_dl.dataset, shuffle=True)
        train_dl = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dl.collate_fn,
        )

    # --- Speaker permutations for ATS/PIL ---
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets, get_pil_targets
    speaker_inds = list(range(MAX_NUM_SPKS))
    all_perms = list(itertools.permutations(speaker_inds))
    max_perms = 2000
    if len(all_perms) > max_perms:
        identity = [tuple(speaker_inds)]
        others = [p for p in all_perms if p != tuple(speaker_inds)]
        sampled = random.sample(others, min(max_perms - 1, len(others)))
        all_perms = identity + sampled
    speaker_permutations = torch.tensor(all_perms)

    pil_weight = PIL_WEIGHT / (PIL_WEIGHT + ATS_WEIGHT)
    ats_weight = ATS_WEIGHT / (PIL_WEIGHT + ATS_WEIGHT)

    # --- DDP model wrapping ---
    if use_ddp:
        nemo_model = DDP(nemo_model, device_ids=[local_rank], find_unused_parameters=True)
        if hasattr(nemo_model, "_set_static_graph"):
            nemo_model._set_static_graph()
        if is_main:
            print(f"Using {dist.get_world_size()} GPUs with DDP")
    else:
        if is_main:
            print("Using 1 GPU")

    # --- Training loop ---
    base_model = nemo_model.module if use_ddp else nemo_model
    if is_main:
        print(f"\nStarting training for {FIXED_STEPS} steps...")
        print("=" * 60)
    nemo_model.train()
    if FREEZE_ENCODER:
        base_model.encoder.eval()
        base_model.preprocessor.eval()

    t_train_start = time.time()
    step = 0
    epoch = 0
    smooth_loss = 0.0
    if use_ddp:
        train_sampler.set_epoch(epoch)
    data_iter = iter(train_dl)

    while step < FIXED_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if use_ddp:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_dl)
            batch = next(data_iter)

        audio_signal, audio_signal_length, targets, target_lens = batch
        audio_signal = audio_signal.to(device)
        audio_signal_length = audio_signal_length.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)

        preds = nemo_model.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            targets = targets[:, :preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=speaker_permutations)

        ats_loss = custom_loss(preds, targets_ats, target_lens)
        pil_loss = custom_loss(preds, targets_pil, target_lens)
        sm_ref = nemo_model.module.sortformer_modules if use_ddp else nemo_model.sortformer_modules
        dec_loss = speaker_output_decorrelation_loss(sm_ref)
        loss = ats_weight * ats_loss + pil_weight * pil_loss + DECORR_WEIGHT * dec_loss

        if torch.isnan(loss) or loss.item() > 100:
            print(f"\nFAIL: loss={loss.item():.4f} at step {step}")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in nemo_model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        ema = 0.9
        smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
        debiased = smooth_loss / (1 - ema ** (step + 1))

        if is_main and (step % 10 == 0 or step == FIXED_STEPS - 1):
            lr_now = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_train_start
            remaining = elapsed / max(step + 1, 1) * (FIXED_STEPS - step - 1)
            print(
                f"\rstep {step:04d}/{FIXED_STEPS} | loss: {debiased:.6f} | "
                f"ats: {ats_loss.item():.4f} | pil: {pil_loss.item():.4f} | "
                f"dec: {dec_loss.item():.4f} | "
                f"lr: {lr_now:.2e} | epoch: {epoch} | "
                f"remaining: {remaining:.0f}s    ",
                end="", flush=True,
            )

        step += 1

        if step == 1:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if is_main:
        print()
    t_train_end = time.time()
    training_seconds = t_train_end - t_train_start

    # --- Non-main processes exit after training ---
    if use_ddp and not is_main:
        dist.destroy_process_group()
        sys.exit(0)
    if use_ddp:
        dist.destroy_process_group()

    # --- Evaluation (main process only) ---
    if is_main:
        print("\n" + "=" * 60)
        print("Evaluating on sampled multi-dataset...")
        eval_model = base_model
        eval_model.eval()

        from prepare import STREAMING_CHUNK_LEN, STREAMING_CHUNK_RC, STREAMING_FIFO_LEN, STREAMING_SPKCACHE_UPDATE
        eval_model.sortformer_modules.chunk_len = STREAMING_CHUNK_LEN
        eval_model.sortformer_modules.chunk_right_context = STREAMING_CHUNK_RC
        eval_model.sortformer_modules.fifo_len = STREAMING_FIFO_LEN
        eval_model.sortformer_modules.spkcache_update_period = STREAMING_SPKCACHE_UPDATE

        metrics = evaluate_der(eval_model)

        # --- Final summary ---
        t_end = time.time()
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        print("\n---")
        print(f"der:              {metrics['DER']:.6f}")
        print(f"fa:               {metrics['FA']:.6f}")
        print(f"miss:             {metrics['MISS']:.6f}")
        print(f"cer:              {metrics['CER']:.6f}")
        print(f"spk_count_acc:    {metrics['Spk_Count_Acc']:.6f}")
        print(f"n_sessions:       {metrics['n_sessions']}")
        print(f"n_datasets:       {metrics['n_datasets']}")
        print(f"training_steps:   {FIXED_STEPS}")
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
        print(f"trainable_params: {trainable}")
        print(f"src_spks:         {SRC_NUM_SPKS}")
        print(f"dst_spks:         {NUM_SPKS}")
