"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

VAE decoder using diffusers' AutoencoderKLWan with custom weight mapping.
"""
import torch
import safetensors.torch as st
from diffusers.models import AutoencoderKLWan


def build_key_mapping():
    """Build mapping from safetensors keys to diffusers AutoencoderKLWan keys."""
    mapping = {}

    # Top-level conv
    mapping["conv1.weight"] = "quant_conv.weight"
    mapping["conv1.bias"] = "quant_conv.bias"
    mapping["conv2.weight"] = "post_quant_conv.weight"
    mapping["conv2.bias"] = "post_quant_conv.bias"

    # Decoder conv_in
    mapping["decoder.conv1.weight"] = "decoder.conv_in.weight"
    mapping["decoder.conv1.bias"] = "decoder.conv_in.bias"

    # Decoder norm_out + conv_out (head)
    mapping["decoder.head.0.gamma"] = "decoder.norm_out.gamma"
    mapping["decoder.head.2.weight"] = "decoder.conv_out.weight"
    mapping["decoder.head.2.bias"] = "decoder.conv_out.bias"

    # Middle block: 2 resnets + 1 attention
    for file_idx, model_idx in [(0, 0), (2, 1)]:
        fp = f"decoder.middle.{file_idx}.residual"
        mp = f"decoder.mid_block.resnets.{model_idx}"
        mapping[f"{fp}.0.gamma"] = f"{mp}.norm1.gamma"
        mapping[f"{fp}.2.weight"] = f"{mp}.conv1.weight"
        mapping[f"{fp}.2.bias"] = f"{mp}.conv1.bias"
        mapping[f"{fp}.3.gamma"] = f"{mp}.norm2.gamma"
        mapping[f"{fp}.6.weight"] = f"{mp}.conv2.weight"
        mapping[f"{fp}.6.bias"] = f"{mp}.conv2.bias"

    # Middle attention
    mapping["decoder.middle.1.norm.gamma"] = "decoder.mid_block.attentions.0.norm.gamma"
    mapping["decoder.middle.1.to_qkv.weight"] = "decoder.mid_block.attentions.0.to_qkv.weight"
    mapping["decoder.middle.1.to_qkv.bias"] = "decoder.mid_block.attentions.0.to_qkv.bias"
    mapping["decoder.middle.1.proj.weight"] = "decoder.mid_block.attentions.0.proj.weight"
    mapping["decoder.middle.1.proj.bias"] = "decoder.mid_block.attentions.0.proj.bias"

    # Up blocks mapping
    # File: flat upsamples[0..14]
    # Model: up_blocks[0..3] with resnets and upsamplers
    upsample_map = [
        # (file_idx, block_idx, type, sub_idx)
        (0, 0, "resnet", 0), (1, 0, "resnet", 1), (2, 0, "resnet", 2),
        (3, 0, "upsampler", 0),
        (4, 1, "resnet", 0), (5, 1, "resnet", 1), (6, 1, "resnet", 2),
        (7, 1, "upsampler", 0),
        (8, 2, "resnet", 0), (9, 2, "resnet", 1), (10, 2, "resnet", 2),
        (11, 2, "upsampler", 0),
        (12, 3, "resnet", 0), (13, 3, "resnet", 1), (14, 3, "resnet", 2),
    ]

    for file_idx, block_idx, typ, sub_idx in upsample_map:
        fp = f"decoder.upsamples.{file_idx}"
        if typ == "resnet":
            mp = f"decoder.up_blocks.{block_idx}.resnets.{sub_idx}"
            mapping[f"{fp}.residual.0.gamma"] = f"{mp}.norm1.gamma"
            mapping[f"{fp}.residual.2.weight"] = f"{mp}.conv1.weight"
            mapping[f"{fp}.residual.2.bias"] = f"{mp}.conv1.bias"
            mapping[f"{fp}.residual.3.gamma"] = f"{mp}.norm2.gamma"
            mapping[f"{fp}.residual.6.weight"] = f"{mp}.conv2.weight"
            mapping[f"{fp}.residual.6.bias"] = f"{mp}.conv2.bias"
            # Shortcut if present
            if file_idx == 4:  # upsamples.4 has shortcut
                mapping[f"{fp}.shortcut.weight"] = f"{mp}.conv_shortcut.weight"
                mapping[f"{fp}.shortcut.bias"] = f"{mp}.conv_shortcut.bias"
        elif typ == "upsampler":
            mp = f"decoder.up_blocks.{block_idx}.upsamplers.0"
            mapping[f"{fp}.resample.1.weight"] = f"{mp}.resample.1.weight"
            mapping[f"{fp}.resample.1.bias"] = f"{mp}.resample.1.bias"
            # time_conv only for blocks 0 and 1 (temporal downsample=[False, True, True])
            if block_idx < 2:
                mapping[f"{fp}.time_conv.weight"] = f"{mp}.time_conv.weight"
                mapping[f"{fp}.time_conv.bias"] = f"{mp}.time_conv.bias"

    # Encoder mappings (same pattern)
    mapping["encoder.conv1.weight"] = "encoder.conv_in.weight"
    mapping["encoder.conv1.bias"] = "encoder.conv_in.bias"
    mapping["encoder.head.0.gamma"] = "encoder.norm_out.gamma"
    mapping["encoder.head.2.weight"] = "encoder.conv_out.weight"
    mapping["encoder.head.2.bias"] = "encoder.conv_out.bias"

    # Encoder middle
    for file_idx, model_idx in [(0, 0), (2, 1)]:
        fp = f"encoder.middle.{file_idx}.residual"
        mp = f"encoder.mid_block.resnets.{model_idx}"
        mapping[f"{fp}.0.gamma"] = f"{mp}.norm1.gamma"
        mapping[f"{fp}.2.weight"] = f"{mp}.conv1.weight"
        mapping[f"{fp}.2.bias"] = f"{mp}.conv1.bias"
        mapping[f"{fp}.3.gamma"] = f"{mp}.norm2.gamma"
        mapping[f"{fp}.6.weight"] = f"{mp}.conv2.weight"
        mapping[f"{fp}.6.bias"] = f"{mp}.conv2.bias"

    mapping["encoder.middle.1.norm.gamma"] = "encoder.mid_block.attentions.0.norm.gamma"
    mapping["encoder.middle.1.to_qkv.weight"] = "encoder.mid_block.attentions.0.to_qkv.weight"
    mapping["encoder.middle.1.to_qkv.bias"] = "encoder.mid_block.attentions.0.to_qkv.bias"
    mapping["encoder.middle.1.proj.weight"] = "encoder.mid_block.attentions.0.proj.weight"
    mapping["encoder.middle.1.proj.bias"] = "encoder.mid_block.attentions.0.proj.bias"

    # Encoder down blocks
    downsample_map = [
        (0, 0, "resnet", 0), (1, 0, "resnet", 1),
        (2, 0, "downsampler", 0),
        (3, 1, "resnet", 0), (4, 1, "resnet", 1),
        (5, 1, "downsampler", 0),
        (6, 2, "resnet", 0), (7, 2, "resnet", 1),
        (8, 2, "downsampler", 0),
        (9, 3, "resnet", 0), (10, 3, "resnet", 1),
    ]

    for file_idx, block_idx, typ, sub_idx in downsample_map:
        fp = f"encoder.downsamples.{file_idx}"
        if typ == "resnet":
            mp = f"encoder.down_blocks.{block_idx}.resnets.{sub_idx}"
            mapping[f"{fp}.residual.0.gamma"] = f"{mp}.norm1.gamma"
            mapping[f"{fp}.residual.2.weight"] = f"{mp}.conv1.weight"
            mapping[f"{fp}.residual.2.bias"] = f"{mp}.conv1.bias"
            mapping[f"{fp}.residual.3.gamma"] = f"{mp}.norm2.gamma"
            mapping[f"{fp}.residual.6.weight"] = f"{mp}.conv2.weight"
            mapping[f"{fp}.residual.6.bias"] = f"{mp}.conv2.bias"
            if file_idx in [3, 6]:  # encoder shortcut
                mapping[f"{fp}.shortcut.weight"] = f"{mp}.conv_shortcut.weight"
                mapping[f"{fp}.shortcut.bias"] = f"{mp}.conv_shortcut.bias"
        elif typ == "downsampler":
            mp = f"encoder.down_blocks.{block_idx}.downsamplers.0"
            mapping[f"{fp}.resample.1.weight"] = f"{mp}.resample.1.weight"
            mapping[f"{fp}.resample.1.bias"] = f"{mp}.resample.1.bias"
            if block_idx >= 1:
                mapping[f"{fp}.time_conv.weight"] = f"{mp}.time_conv.weight"
                mapping[f"{fp}.time_conv.bias"] = f"{mp}.time_conv.bias"

    return mapping


def load_vae(safetensors_path, device="cuda", dtype=torch.bfloat16):
    """Load VAE from safetensors with weight remapping."""
    # Create model
    vae = AutoencoderKLWan(
        base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True],
    )

    # Load weights
    raw_sd = st.load_file(safetensors_path, device="cpu")
    key_mapping = build_key_mapping()

    # Remap
    new_sd = {}
    unmapped = []
    for file_key, tensor in raw_sd.items():
        model_key = key_mapping.get(file_key)
        if model_key is not None:
            new_sd[model_key] = tensor
        else:
            unmapped.append(file_key)

    if unmapped:
        print(f"Warning: {len(unmapped)} unmapped keys: {unmapped[:5]}")

    # Check for missing keys
    model_sd = vae.state_dict()
    missing = set(model_sd.keys()) - set(new_sd.keys())
    extra = set(new_sd.keys()) - set(model_sd.keys())
    if missing:
        print(f"Missing keys: {len(missing)}: {list(missing)[:5]}")
    if extra:
        print(f"Extra keys: {len(extra)}: {list(extra)[:5]}")

    vae.load_state_dict(new_sd, strict=False)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae
