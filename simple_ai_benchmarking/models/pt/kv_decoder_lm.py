# Project Name: simple-ai-benchmarking
# File Name: kv_decoder_lm.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# A layer's KV cache entry: (keys, values), each (batch, heads, seq, head_dim).
LayerKV = Tuple[torch.Tensor, torch.Tensor]
KVCache = List[LayerKV]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability, then return to the input dtype so the
        # rest of the network runs at the requested (lower) precision.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim).
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with RoPE and an optional KV cache.

    Prefill passes the whole prompt (is_causal=True). Decode passes a single
    token whose query attends to every cached key/value, so no causal mask is
    needed for that step."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[LayerKV] = None,
    ) -> Tuple[torch.Tensor, LayerKV]:
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_kv = (k, v)

        # Causal masking only matters during prefill (multiple new queries). For a
        # single decode token the query legitimately attends to all cached keys.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.num_heads * self.head_dim)
        return self.wo(out), new_kv


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, feedforward_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[LayerKV] = None,
    ) -> Tuple[torch.Tensor, LayerKV]:
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


class KVCacheDecoderLM(nn.Module):
    """Decoder-only transformer with RoPE, RMSNorm, SwiGLU and a real KV cache.

    Unlike SimpleTransformerLanguageModel (which recomputes the full sequence on
    every step), this keeps per-layer key/value caches so decode is O(1) per
    token. That makes time-to-first-token (prefill) and decode throughput
    measure what they do on a production serving stack."""

    def __init__(
        self,
        vocab_size: int = 32000,
        context_length: int = 4096,
        embedding_dim: int = 2048,
        num_heads: int = 16,
        num_layers: int = 16,
        feedforward_dim: int = 5632,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.head_dim = embedding_dim // num_heads
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList(
            DecoderBlock(embedding_dim, num_heads, feedforward_dim)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        inv_freq = 1.0 / (
            rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(
        self, start: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(start, start + length, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self, token_ids: torch.Tensor, past_kvs: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, KVCache]:
        _, seq = token_ids.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        hidden = self.token_embedding(token_ids)
        cos, sin = self._rope(past_len, seq, hidden.device, hidden.dtype)

        new_cache: KVCache = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past_kvs is None else past_kvs[index]
            hidden, layer_kv = layer(hidden, cos, sin, layer_past)
            new_cache.append(layer_kv)

        logits = self.lm_head(self.norm(hidden))
        return logits, new_cache

    @torch.no_grad()
    def prefill(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
        """Process the prompt once and return (next_token, kv_cache)."""
        logits, cache = self.forward(token_ids[:, -self.context_length :])
        next_token = logits[:, -1:, :].argmax(dim=-1)
        return next_token, cache

    @torch.no_grad()
    def decode_step(
        self, token: torch.Tensor, past_kvs: KVCache
    ) -> Tuple[torch.Tensor, KVCache]:
        """Advance one token using the cache (no full-sequence recompute)."""
        logits, cache = self.forward(token, past_kvs)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        return next_token, cache

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, generated_tokens: int) -> torch.Tensor:
        """Greedy generation using the KV cache (returns the full sequence)."""
        self.eval()
        next_token, cache = self.prefill(token_ids)
        out = torch.cat((token_ids, next_token), dim=1)
        for _ in range(max(0, generated_tokens - 1)):
            next_token, cache = self.decode_step(next_token, cache)
            out = torch.cat((out, next_token), dim=1)
        return out
