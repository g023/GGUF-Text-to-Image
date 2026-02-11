"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Qwen-Image GGUF Text-to-Image Generation Pipeline.
Standalone Python script - no ComfyUI dependency.
Uses CUDA with sequential loading for 12GB GPU.
Supports batch processing of multiple prompts in parallel.
"""
import argparse
import math
import time
import torch
import numpy as np
from PIL import Image

from text_encoder import Qwen2TextEncoder
from dit import FluxDiT, rms_norm
from vae import load_vae
from plugins import PluginRegistry


def get_tokenizer():
    """Load Qwen2.5 tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def patchify(latent, patch_size=2):
    """Convert latent [B, C, H, W] to patches [B, num_patches, patch_dim].
    patch_dim = C * patch_size * patch_size
    """
    B, C, H, W = latent.shape
    pH = H // patch_size
    pW = W // patch_size
    x = latent.reshape(B, C, pH, patch_size, pW, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, pH * pW, C * patch_size * patch_size)
    return x, pH, pW


def unpatchify(patches, pH, pW, channels=16, patch_size=2):
    """Convert patches [B, num_patches, patch_dim] back to [B, C, H, W]."""
    B = patches.shape[0]
    x = patches.reshape(B, pH, pW, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, channels, pH * patch_size, pW * patch_size)
    return x


@torch.no_grad()
def generate_image(
    prompts,
    height=512,
    width=512,
    num_steps=20,
    guidance_scale=1.0,
    seeds=None,
    device="cuda",
    dtype=torch.bfloat16,
    dit_path="GGUF/Qwen-Rapid-NSW-v23_Q3_K.gguf",
    te_path="GGUF/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
    vae_path="GGUF/qwen_image_vae.safetensors",
    sampler_name="euler",
    scheduler_name="flow_match",
    prompt_processor_name="qwen_chat",
):
    """Full text-to-image generation pipeline with batch support.

    Args:
        prompts: Single prompt string or list of prompt strings.
        seeds: Single seed int, list of seed ints, or None.
    Returns:
        List of PIL Images.
    """
    total_start = time.time()

    # Normalize inputs
    if isinstance(prompts, str):
        prompts = [prompts]
    batch_size = len(prompts)
    if seeds is None:
        seeds = list(range(42, 42 + batch_size))
    elif isinstance(seeds, int):
        seeds = [seeds + i for i in range(batch_size)]
    if len(seeds) != batch_size:
        raise ValueError(f"Number of seeds ({len(seeds)}) must match number of prompts ({batch_size})")

    # Load plugins
    sampler = PluginRegistry.get('sampler', sampler_name)
    scheduler = PluginRegistry.get('scheduler', scheduler_name)
    prompt_processor = PluginRegistry.get('prompt_processor', prompt_processor_name)

    # Latent dimensions
    latent_channels = 16
    patch_size = 2
    vae_scale_factor = 8
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    print(f"Generating {batch_size}x {width}x{height} images ({latent_w}x{latent_h} latent)")
    print(f"Steps: {num_steps}, Seeds: {seeds}")
    print(f"Sampler: {sampler_name}, Scheduler: {scheduler_name}, Prompt: {prompt_processor_name}")

    # ============ STEP 1: Text Encoding ============
    print("\n[Step 1/3] Text Encoding...")
    step_start = time.time()

    tokenizer = get_tokenizer()
    text_encoder = Qwen2TextEncoder(te_path, device=device, dtype=dtype)

    # Encode each prompt individually for correctness
    txt_hiddens = []
    for i, prompt in enumerate(prompts):
        input_ids, drop_idx = prompt_processor.process(prompt, tokenizer)
        hidden = text_encoder.encode(input_ids)
        hidden = hidden[:, drop_idx:]
        hidden = hidden.to(dtype)
        txt_hiddens.append(hidden)
        print(f"  Prompt {i}: \"{prompt[:50]}{'...' if len(prompt)>50 else ''}\" -> {hidden.shape[1]} tokens")

    # Pad to max length and stack into batch
    max_txt_len = max(h.shape[1] for h in txt_hiddens)
    padded = []
    for h in txt_hiddens:
        if h.shape[1] < max_txt_len:
            pad = torch.zeros(1, max_txt_len - h.shape[1], h.shape[2],
                              device=device, dtype=dtype)
            h = torch.cat([h, pad], dim=1)
        padded.append(h)
    txt_hidden = torch.cat(padded, dim=0)  # [batch, max_txt_len, 3584]

    print(f"  Batched text embeddings: {txt_hidden.shape}")
    print(f"  Text encoding took {time.time()-step_start:.1f}s")

    # Offload text encoder
    del text_encoder
    torch.cuda.empty_cache()
    print(f"  GPU after offload: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    # ============ STEP 2: Denoising ============
    print("\n[Step 2/3] Denoising...")
    step_start = time.time()

    # Generate noise for each seed
    latent_noises = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, latent_channels, latent_h, latent_w,
                            device=device, dtype=dtype, generator=generator)
        latent_noises.append(noise)
    latent_noise = torch.cat(latent_noises, dim=0)  # [batch, C, H, W]

    # Patchify
    img_patches, pH, pW = patchify(latent_noise, patch_size)
    print(f"  Latent patches: {img_patches.shape} (grid: {pH}x{pW})")

    # Load DiT
    dit = FluxDiT(dit_path, device=device, dtype=dtype)

    # Sigma schedule via plugin
    image_seq_len = pH * pW
    sigmas = scheduler.get_sigmas(num_steps, image_seq_len=image_seq_len)
    mu_info = f", mu={scheduler.get_mu(image_seq_len):.3f}" if hasattr(scheduler, 'get_mu') else ""
    print(f"  Schedule: {num_steps} steps{mu_info} ({scheduler_name})")

    # Denoising loop
    latent = img_patches
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        t_batch = torch.full((batch_size,), sigma, device=device, dtype=dtype)

        # Predict velocity
        v_pred = dit.forward(latent, txt_hidden, t_batch, pH, pW)

        # Sampler step via plugin
        latent = sampler.step(v_pred, sigma, sigma_next, latent)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Step {i+1}/{num_steps}: sigma={sigma:.4f}->{sigma_next:.4f}")
        torch.cuda.empty_cache()

    print(f"  Denoising took {time.time()-step_start:.1f}s")

    # Offload DiT
    del dit
    torch.cuda.empty_cache()
    print(f"  GPU after offload: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    # ============ STEP 3: VAE Decode ============
    print("\n[Step 3/3] VAE Decoding...")
    step_start = time.time()

    # Unpatchify
    latent_img = unpatchify(latent, pH, pW, channels=latent_channels, patch_size=patch_size)
    print(f"  Unpatchified latent: {latent_img.shape}")

    # Load VAE
    vae = load_vae(vae_path, device=device, dtype=dtype)

    # VAE expects [B, C, T, H, W] for 3D VAE (T=1 for image)
    latent_img = latent_img.unsqueeze(2)

    # Denormalize latents
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
    latent_img = latent_img * latents_std + latents_mean
    print(f"  VAE input: {latent_img.shape} (denormalized)")

    # Decode
    decoded = vae.decode(latent_img).sample
    print(f"  VAE output: {decoded.shape}")
    print(f"  VAE decoding took {time.time()-step_start:.1f}s")

    # Offload VAE
    del vae
    torch.cuda.empty_cache()

    # Convert to images
    decoded = decoded.squeeze(2)  # Remove T dim: [B, 3, H, W]
    images = []
    for b in range(batch_size):
        img_tensor = decoded[b]  # [3, H, W]
        img_tensor = img_tensor.float().clamp(-1, 1)
        img_tensor = (img_tensor + 1) / 2  # [-1,1] -> [0,1]
        img_array = (img_tensor * 255).byte().cpu().numpy()
        img_array = img_array.transpose(1, 2, 0)  # CHW -> HWC
        images.append(Image.fromarray(img_array))

    total_time = time.time() - total_start
    print(f"\nTotal generation time: {total_time:.1f}s ({total_time/batch_size:.1f}s/image)")

    return images


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image GGUF Text-to-Image Generator")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single text prompt (backward compatible)")
    parser.add_argument("--prompts", nargs='+', type=str, default=None,
                        help="Multiple text prompts for batch generation")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (single)")
    parser.add_argument("--seeds", nargs='+', type=int, default=None,
                        help="Random seeds (one per prompt)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--sampler", type=str, default="euler",
                        help="Sampler plugin (euler, euler_a)")
    parser.add_argument("--scheduler", type=str, default="flow_match",
                        help="Scheduler plugin (flow_match, linear, karras)")
    parser.add_argument("--prompt-processor", type=str, default="qwen_chat",
                        help="Prompt processor plugin (qwen_chat, raw)")
    args = parser.parse_args()

    # Resolve prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = ["a beautiful sunset over the ocean, highly detailed, 4k photograph"]

    # Resolve seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed + i for i in range(len(prompts))]
    else:
        seeds = None

    images = generate_image(
        prompts=prompts,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        seeds=seeds,
        sampler_name=args.sampler,
        scheduler_name=args.scheduler,
        prompt_processor_name=args.prompt_processor,
    )

    # Save images
    base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'png')
    if len(images) == 1:
        images[0].save(args.output)
        print(f"Image saved to {args.output}")
    else:
        for i, image in enumerate(images):
            path = f"{base}_{i}.{ext}"
            image.save(path)
            print(f"Image {i} saved to {path}")


if __name__ == "__main__":
    main()
