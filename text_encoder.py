"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Minimal Qwen2.5 text encoder with on-the-fly GGUF dequantization.
Processes layer-by-layer to fit in 12GB GPU.
"""
import torch
import torch.nn.functional as F
import math
from gguf_utils import load_gguf_lazy, lazy_dequant

# LLAMA_SD_MAP: maps GGUF llama.cpp names to HF-style names
LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}


def remap_key(name):
    for src, dst in LLAMA_SD_MAP.items():
        name = name.replace(src, dst)
    return name


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(orig_dtype)


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def build_rope_cache(seq_len, head_dim, base=1000000.0, device="cuda", dtype=torch.float16):
    """Build rotary position embedding cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


class Qwen2TextEncoder:
    """Minimal Qwen2.5 text encoder using on-the-fly GGUF dequantization."""

    def __init__(self, gguf_path, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        print(f"Loading text encoder from {gguf_path}...")
        self.tensors, self.metadata = load_gguf_lazy(gguf_path)

        # Extract config from metadata
        self.hidden_size = int(self.metadata.get("qwen2vl.embedding_length", 3584))
        self.num_layers = int(self.metadata.get("qwen2vl.block_count", 28))
        self.num_heads = int(self.metadata.get("qwen2vl.attention.head_count", 28))
        self.num_kv_heads = int(self.metadata.get("qwen2vl.attention.head_count_kv", 4))
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(self.metadata.get("qwen2vl.feed_forward_length", 18944))
        self.rope_base = float(self.metadata.get("qwen2vl.rope.freq_base", 1000000.0))
        self.rms_eps = float(self.metadata.get("qwen2vl.attention.layer_norm_rms_epsilon", 1e-6))
        self.vocab_size = int(self.metadata.get("qwen2vl.vocab_size", 152064))

        print(f"  Config: layers={self.num_layers}, hidden={self.hidden_size}, "
              f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
              f"intermediate={self.intermediate_size}")

    def get_weight(self, name, device=None, dtype=None):
        """Get a dequantized weight tensor."""
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return lazy_dequant(self.tensors[name], dtype=dtype, device=device)

    def forward_layer(self, hidden_states, layer_idx, cos, sin):
        """Forward pass through one transformer layer."""
        prefix = f"blk.{layer_idx}."

        # Input layernorm
        norm_weight = self.get_weight(prefix + "attn_norm.weight")
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, norm_weight, self.rms_eps)
        del norm_weight

        # Self attention
        bsz, seq_len, _ = hidden_states.shape

        q = F.linear(hidden_states, self.get_weight(prefix + "attn_q.weight"),
                      self.get_weight(prefix + "attn_q.bias"))
        k = F.linear(hidden_states, self.get_weight(prefix + "attn_k.weight"),
                      self.get_weight(prefix + "attn_k.bias"))
        v = F.linear(hidden_states, self.get_weight(prefix + "attn_v.weight"),
                      self.get_weight(prefix + "attn_v.bias"))

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos_seq = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_seq = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_emb(q, k, cos_seq, sin_seq)

        # GQA: repeat k/v heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

        # Output projection
        attn_output = F.linear(attn_output, self.get_weight(prefix + "attn_output.weight"))
        hidden_states = residual + attn_output

        # Post-attention layernorm + MLP
        residual = hidden_states
        norm_weight = self.get_weight(prefix + "ffn_norm.weight")
        hidden_states = rms_norm(hidden_states, norm_weight, self.rms_eps)
        del norm_weight

        gate = F.linear(hidden_states, self.get_weight(prefix + "ffn_gate.weight"))
        up = F.linear(hidden_states, self.get_weight(prefix + "ffn_up.weight"))
        hidden_states = F.silu(gate) * up
        del gate, up
        hidden_states = F.linear(hidden_states, self.get_weight(prefix + "ffn_down.weight"))
        hidden_states = residual + hidden_states

        return hidden_states

    @torch.no_grad()
    def encode(self, input_ids):
        """
        Encode text tokens to hidden states.
        input_ids: [batch, seq_len] tensor of token IDs
        Returns: [batch, seq_len, hidden_size] hidden states
        """
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        # Token embeddings - dequantize on CPU first (large tensor)
        embed_weight = self.get_weight("token_embd.weight", device="cpu", dtype=self.dtype)
        hidden_states = F.embedding(input_ids.cpu(), embed_weight).to(self.device)
        del embed_weight
        torch.cuda.empty_cache()

        # Build RoPE cache
        cos, sin = build_rope_cache(seq_len, self.head_dim, self.rope_base,
                                    device=self.device, dtype=self.dtype)

        # Process layers sequentially
        for i in range(self.num_layers):
            hidden_states = self.forward_layer(hidden_states, i, cos, sin)
            torch.cuda.empty_cache()

        # Final layernorm
        norm_weight = self.get_weight("output_norm.weight")
        hidden_states = rms_norm(hidden_states, norm_weight, self.rms_eps)
        del norm_weight

        return hidden_states

    def offload(self):
        """Clear GPU memory."""
        torch.cuda.empty_cache()
