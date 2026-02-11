"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

QwenImage DiT (Double-stream Diffusion Transformer) with on-the-fly GGUF dequantization.
Architecture matches diffusers QwenImageTransformer2DModel exactly.
"""
import torch
import torch.nn.functional as F
import math
import functools
from gguf_utils import load_gguf_lazy, lazy_dequant


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                           downscale_freq_shift=1, scale=1, max_period=10000):
    """Matches diffusers get_timestep_embedding exactly."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    return emb


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(orig_dtype)


def apply_rotary_emb_complex(x, freqs_cis):
    """Apply rotary embeddings using complex number rotation (QwenImage style).
    x: [B, S, H, D], freqs_cis: [S, D/2] complex tensor
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1).to(x.device)  # [S, 1, D/2]
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope:
    """Matches diffusers QwenEmbedRope exactly. Uses complex frequencies with scale_rope."""

    def __init__(self, theta=10000, axes_dim=(16, 56, 56), scale_rope=True):
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.pos_freqs = torch.cat([
            self._rope_params(pos_index, axes_dim[0], theta),
            self._rope_params(pos_index, axes_dim[1], theta),
            self._rope_params(pos_index, axes_dim[2], theta),
        ], dim=1)

        self.neg_freqs = torch.cat([
            self._rope_params(neg_index, axes_dim[0], theta),
            self._rope_params(neg_index, axes_dim[1], theta),
            self._rope_params(neg_index, axes_dim[2], theta),
        ], dim=1)

    @staticmethod
    def _rope_params(index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            index.float(),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx:idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][:height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][:width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)
        return freqs.clone().contiguous()

    def __call__(self, video_fhw, txt_seq_len, device):
        """
        video_fhw: (frame, height, width) tuple
        txt_seq_len: int, number of text tokens
        Returns: (img_freqs, txt_freqs) each complex [S, D/2]
        """
        frame, height, width = video_fhw
        img_freqs = self._compute_video_freqs(frame, height, width, idx=0).to(device)

        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)

        txt_freqs = self.pos_freqs[max_vid_index:max_vid_index + txt_seq_len].to(device)
        return img_freqs, txt_freqs


class QwenImageDiT:
    """QwenImage Double-stream DiT with on-the-fly GGUF dequantization."""

    def __init__(self, gguf_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self._weight_cache = {}

        print(f"Loading DiT from {gguf_path}...")
        self.tensors, self.metadata = load_gguf_lazy(gguf_path)

        self.hidden_size = 3072
        self.num_heads = 24
        self.head_dim = 128
        self.num_double_blocks = 60
        self.text_dim = 3584
        self.img_dim = 64
        self.axes_dims_rope = (16, 56, 56)

        self.rope = QwenEmbedRope(theta=10000, axes_dim=self.axes_dims_rope, scale_rope=True)

        print(f"  Config: blocks={self.num_double_blocks}, hidden={self.hidden_size}, "
              f"heads={self.num_heads}, head_dim={self.head_dim}")

    def get_weight(self, name, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if name not in self.tensors:
            return None
        # Cache small weights (biases, norm weights) to avoid repeated dequantization
        cache_key = (name, device, dtype)
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]
        weight = lazy_dequant(self.tensors[name], dtype=dtype, device=device)
        if weight.numel() <= 16384:  # Cache tensors <= 32KB (bf16)
            self._weight_cache[cache_key] = weight
        return weight

    def forward_block(self, img, txt, temb, block_idx, img_freqs, txt_freqs):
        """Forward through one double-stream transformer block."""
        prefix = f"transformer_blocks.{block_idx}."

        # === Modulation (chunk into 2 halves, then each into shift/scale/gate) ===
        img_mod = F.linear(F.silu(temb),
                           self.get_weight(prefix + "img_mod.1.weight"),
                           self.get_weight(prefix + "img_mod.1.bias"))
        img_mod1, img_mod2 = img_mod.chunk(2, dim=-1)
        img_shift1, img_scale1, img_gate1 = img_mod1.unsqueeze(1).chunk(3, dim=-1)
        img_shift2, img_scale2, img_gate2 = img_mod2.unsqueeze(1).chunk(3, dim=-1)

        txt_mod = F.linear(F.silu(temb),
                           self.get_weight(prefix + "txt_mod.1.weight"),
                           self.get_weight(prefix + "txt_mod.1.bias"))
        txt_mod1, txt_mod2 = txt_mod.chunk(2, dim=-1)
        txt_shift1, txt_scale1, txt_gate1 = txt_mod1.unsqueeze(1).chunk(3, dim=-1)
        txt_shift2, txt_scale2, txt_gate2 = txt_mod2.unsqueeze(1).chunk(3, dim=-1)

        # === Pre-norm + modulate ===
        img_norm = F.layer_norm(img, (self.hidden_size,))
        img_norm = img_norm * (1 + img_scale1) + img_shift1

        txt_norm = F.layer_norm(txt, (self.hidden_size,))
        txt_norm = txt_norm * (1 + txt_scale1) + txt_shift1

        bsz = img.shape[0]
        img_seq = img.shape[1]
        txt_seq = txt.shape[1]

        # === Q/K/V projections ===
        img_q = F.linear(img_norm, self.get_weight(prefix + "attn.to_q.weight"),
                         self.get_weight(prefix + "attn.to_q.bias"))
        img_k = F.linear(img_norm, self.get_weight(prefix + "attn.to_k.weight"),
                         self.get_weight(prefix + "attn.to_k.bias"))
        img_v = F.linear(img_norm, self.get_weight(prefix + "attn.to_v.weight"),
                         self.get_weight(prefix + "attn.to_v.bias"))
        del img_norm

        txt_q = F.linear(txt_norm, self.get_weight(prefix + "attn.add_q_proj.weight"),
                         self.get_weight(prefix + "attn.add_q_proj.bias"))
        txt_k = F.linear(txt_norm, self.get_weight(prefix + "attn.add_k_proj.weight"),
                         self.get_weight(prefix + "attn.add_k_proj.bias"))
        txt_v = F.linear(txt_norm, self.get_weight(prefix + "attn.add_v_proj.weight"),
                         self.get_weight(prefix + "attn.add_v_proj.bias"))
        del txt_norm

        # Reshape to [B, S, H, D]
        img_q = img_q.view(bsz, img_seq, self.num_heads, self.head_dim)
        img_k = img_k.view(bsz, img_seq, self.num_heads, self.head_dim)
        img_v = img_v.view(bsz, img_seq, self.num_heads, self.head_dim)
        txt_q = txt_q.view(bsz, txt_seq, self.num_heads, self.head_dim)
        txt_k = txt_k.view(bsz, txt_seq, self.num_heads, self.head_dim)
        txt_v = txt_v.view(bsz, txt_seq, self.num_heads, self.head_dim)

        # === QK normalization (RMSNorm) ===
        def qk_norm(x, weight, eps=1e-6):
            rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
            return (x.float() / rms * weight.float().reshape(1, 1, 1, -1)).to(self.dtype)

        img_q = qk_norm(img_q, self.get_weight(prefix + "attn.norm_q.weight"))
        img_k = qk_norm(img_k, self.get_weight(prefix + "attn.norm_k.weight"))
        txt_q = qk_norm(txt_q, self.get_weight(prefix + "attn.norm_added_q.weight"))
        txt_k = qk_norm(txt_k, self.get_weight(prefix + "attn.norm_added_k.weight"))

        # === Apply RoPE SEPARATELY (QwenImage: complex rotation, different freqs) ===
        img_q = apply_rotary_emb_complex(img_q, img_freqs)
        img_k = apply_rotary_emb_complex(img_k, img_freqs)
        txt_q = apply_rotary_emb_complex(txt_q, txt_freqs)
        txt_k = apply_rotary_emb_complex(txt_k, txt_freqs)

        # === Concatenate: TEXT first, IMAGE second ===
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)
        del img_q, img_k, img_v, txt_q, txt_k, txt_v

        # === Joint attention ===
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2).contiguous()
        attn = F.scaled_dot_product_attention(q, k, v)
        del q, k, v

        # === Split: text first, image second ===
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(bsz, txt_seq + img_seq, self.hidden_size)
        txt_attn = attn[:, :txt_seq, :]
        img_attn = attn[:, txt_seq:, :]
        del attn

        # === Output projections ===
        img_attn = F.linear(img_attn, self.get_weight(prefix + "attn.to_out.0.weight"),
                            self.get_weight(prefix + "attn.to_out.0.bias"))
        txt_attn = F.linear(txt_attn, self.get_weight(prefix + "attn.to_add_out.weight"),
                            self.get_weight(prefix + "attn.to_add_out.bias"))

        # === Residual + gate ===
        img = img + img_gate1 * img_attn
        txt = txt + txt_gate1 * txt_attn
        del img_attn, txt_attn

        # === Image MLP ===
        img_norm2 = F.layer_norm(img, (self.hidden_size,))
        img_norm2 = img_norm2 * (1 + img_scale2) + img_shift2
        img_mlp = F.linear(img_norm2, self.get_weight(prefix + "img_mlp.net.0.proj.weight"),
                           self.get_weight(prefix + "img_mlp.net.0.proj.bias"))
        img_mlp = F.gelu(img_mlp, approximate="tanh")
        img_mlp = F.linear(img_mlp, self.get_weight(prefix + "img_mlp.net.2.weight"),
                           self.get_weight(prefix + "img_mlp.net.2.bias"))
        img = img + img_gate2 * img_mlp
        del img_norm2, img_mlp

        # === Text MLP ===
        txt_norm2 = F.layer_norm(txt, (self.hidden_size,))
        txt_norm2 = txt_norm2 * (1 + txt_scale2) + txt_shift2
        txt_mlp = F.linear(txt_norm2, self.get_weight(prefix + "txt_mlp.net.0.proj.weight"),
                           self.get_weight(prefix + "txt_mlp.net.0.proj.bias"))
        txt_mlp = F.gelu(txt_mlp, approximate="tanh")
        txt_mlp = F.linear(txt_mlp, self.get_weight(prefix + "txt_mlp.net.2.weight"),
                           self.get_weight(prefix + "txt_mlp.net.2.bias"))
        txt = txt + txt_gate2 * txt_mlp
        del txt_norm2, txt_mlp

        return img, txt

    @torch.no_grad()
    def forward(self, img_latent, txt_hidden, timestep, img_h, img_w):
        """
        img_latent: [batch, img_seq, 64] - patchified image latents
        txt_hidden: [batch, txt_seq, 3584] - text encoder hidden states
        timestep: [batch] - timestep values (raw, NOT multiplied by 1000)
        img_h, img_w: patch grid dimensions
        Returns: [batch, img_seq, patch_dim] - predicted velocity
        """
        txt_seq = txt_hidden.shape[1]

        # Build RoPE frequencies (QwenImage: separate complex freqs for img and txt)
        img_freqs, txt_freqs = self.rope((1, img_h, img_w), txt_seq, self.device)

        # Text normalization + projection
        txt_norm_weight = self.get_weight("txt_norm.weight")
        txt_hidden = rms_norm(txt_hidden, txt_norm_weight)
        del txt_norm_weight

        txt = F.linear(txt_hidden, self.get_weight("txt_in.weight"),
                       self.get_weight("txt_in.bias"))
        img = F.linear(img_latent, self.get_weight("img_in.weight"),
                       self.get_weight("img_in.bias"))

        # Timestep embedding (QwenImage: flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        t_emb = get_timestep_embedding(
            timestep, 256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        ).to(self.dtype)
        temb = F.linear(F.silu(F.linear(t_emb,
                        self.get_weight("time_text_embed.timestep_embedder.linear_1.weight"),
                        self.get_weight("time_text_embed.timestep_embedder.linear_1.bias"))),
                        self.get_weight("time_text_embed.timestep_embedder.linear_2.weight"),
                        self.get_weight("time_text_embed.timestep_embedder.linear_2.bias"))

        # Process through double-stream blocks
        for i in range(self.num_double_blocks):
            img, txt = self.forward_block(img, txt, temb, i, img_freqs, txt_freqs)
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # Final norm + projection (AdaLayerNormContinuous: scale first, then shift)
        norm_out = F.linear(F.silu(temb),
                            self.get_weight("norm_out.linear.weight"),
                            self.get_weight("norm_out.linear.bias"))
        scale, shift = norm_out.unsqueeze(1).chunk(2, dim=-1)

        img = F.layer_norm(img, (self.hidden_size,))
        img = img * (1 + scale) + shift

        output = F.linear(img, self.get_weight("proj_out.weight"),
                          self.get_weight("proj_out.bias"))

        return output

    def offload(self):
        self._weight_cache.clear()
        torch.cuda.empty_cache()


# Backward compatibility alias
FluxDiT = QwenImageDiT
