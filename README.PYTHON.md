# GGUF-Text-to-Image: GGUF-Powered Text-to-Image Generation

**GGUF-Text-to-Image** is a high-performance, standalone Python library for text-to-image generation using GGUF-quantized diffusion models. Built for efficiency on consumer GPUs, it provides both command-line and programmatic interfaces for generating images from text prompts. Currently ONLY tested working with Qwen-Image GGUF models.

## License

Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Overview

GGUF-Text-to-Image implements a complete text-to-image pipeline with the following components:

- **Text Encoding**: Qwen2.5 language model for rich text understanding
- **Diffusion Transformer**: Efficient DiT architecture with GGUF quantization
- **VAE Decoding**: High-fidelity image reconstruction
- **Plugin System**: Extensible samplers, schedulers, and prompt processors

Key features:
- GGUF quantization for 12GB GPU compatibility
- Batch processing support
- Modular plugin architecture
- Pure Python with minimal dependencies

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (12GB+ VRAM recommended)
- PyTorch with CUDA support

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers safetensors pillow numpy gguf
```

### Setup
1. Place all GGUF-Text-to-Image files in your project directory
2. Ensure model files are in the `GGUF/` subdirectory

## Python API

### Basic Usage

```python
from generate import generate_image

# Generate a single image
images = generate_image(
    prompts="a beautiful sunset over mountains",
    height=512,
    width=512,
    num_steps=25,
    seeds=42
)

# Save the image
images[0].save("sunset.png")
```

### Batch Generation

```python
# Generate multiple images
prompts = [
    "majestic mountain landscape with lake",
    "cyberpunk city street at night",
    "minimalist geometric abstract art"
]

images = generate_image(
    prompts=prompts,
    height=512,
    width=512,
    num_steps=30,
    seeds=[100, 200, 300]
)

# Save all images
for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

### Advanced Configuration

```python
from generate import generate_image

# Custom sampler and scheduler
images = generate_image(
    prompts="professional portrait, studio lighting",
    height=512,
    width=512,
    num_steps=50,
    seeds=123,
    sampler_name="euler_a",      # Ancestral Euler for variety
    scheduler_name="karras",     # Optimized noise schedule
    prompt_processor_name="qwen_chat"  # Chat-formatted prompts
)
```

### Using Components Separately

#### Text Encoding

```python
from text_encoder import Qwen2TextEncoder
from generate import get_tokenizer
from plugins import PluginRegistry

# Load components
tokenizer = get_tokenizer()
text_encoder = Qwen2TextEncoder("GGUF/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf")
prompt_processor = PluginRegistry.get('prompt_processor', 'qwen_chat')

# Process prompt
input_ids, drop_idx = prompt_processor.process("a cute cat", tokenizer)
embeddings = text_encoder.encode(input_ids)
embeddings = embeddings[:, drop_idx:]  # Remove template tokens

print(f"Text embeddings shape: {embeddings.shape}")  # [1, seq_len, 3584]
```

#### Diffusion Model

```python
import torch
from dit import FluxDiT
from generate import patchify, unpatchify

# Load DiT
dit = FluxDiT("GGUF/Qwen-Rapid-NSW-v23_Q3_K.gguf")

# Create latent noise
latent_h, latent_w = 64, 64  # For 512x512 image
noise = torch.randn(1, 16, latent_h, latent_w, device="cuda", dtype=torch.bfloat16)

# Patchify
patches, pH, pW = patchify(noise)

# Mock text embeddings (normally from text encoder)
txt_hidden = torch.randn(1, 77, 3584, device="cuda", dtype=torch.bfloat16)
timestep = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)

# Denoise
velocity = dit.forward(patches, txt_hidden, timestep, pH, pW)

# Unpatchify
latent_out = unpatchify(velocity, pH, pW)
```

#### VAE Decoding

```python
from vae import load_vae

# Load VAE
vae = load_vae("GGUF/qwen_image_vae.safetensors")

# Mock latents (normally from diffusion)
latents = torch.randn(1, 16, 1, 64, 64, device="cuda", dtype=torch.bfloat16)

# Decode
decoded = vae.decode(latents).sample
print(f"Decoded shape: {decoded.shape}")  # [1, 3, 1, 512, 512]
```

### Plugin System

#### Creating Custom Samplers

```python
from plugins import PluginRegistry
from plugins.base import BaseSampler

class CustomSampler(BaseSampler):
    name = "custom"

    def step(self, model_output, sigma, sigma_next, sample):
        # Implement your sampling algorithm
        dt = sigma_next - sigma
        return sample + dt * model_output * 0.9  # Slightly damped

# Register the plugin
PluginRegistry.register('sampler', 'custom')(CustomSampler)

# Use it
images = generate_image(
    prompts="test prompt",
    sampler_name="custom"
)
```

#### Custom Schedulers

```python
from plugins import PluginRegistry
from plugins.base import BaseScheduler
import numpy as np

class ExponentialScheduler(BaseScheduler):
    name = "exponential"

    def get_sigmas(self, num_steps, **kwargs):
        # Exponential decay schedule
        sigmas = np.exp(np.linspace(np.log(1.0), np.log(0.01), num_steps + 1))
        return sigmas

# Register and use
PluginRegistry.register('scheduler', 'exponential')(ExponentialScheduler)
```

#### Custom Prompt Processors

```python
from plugins import PluginRegistry
from plugins.base import BasePromptProcessor

class ArtisticProcessor(BasePromptProcessor):
    name = "artistic"

    def process(self, prompt, tokenizer):
        # Add artistic styling
        styled_prompt = f"Artistic interpretation of: {prompt}, in the style of famous painters"
        input_ids = tokenizer.encode(styled_prompt, return_tensors="pt").squeeze()
        return input_ids, 0  # No tokens to drop

# Register and use
PluginRegistry.register('prompt_processor', 'artistic')(ArtisticProcessor)
```

## API Reference

### generate_image()

Main generation function.

**Parameters:**
- `prompts` (str or list): Text prompt(s)
- `height` (int): Image height (default: 512)
- `width` (int): Image width (default: 512)
- `num_steps` (int): Denoising steps (default: 20)
- `guidance_scale` (float): Classifier-free guidance (default: 1.0)
- `seeds` (int, list, or None): Random seeds
- `device` (str): Compute device (default: "cuda")
- `dtype` (torch.dtype): Data type (default: torch.bfloat16)
- `dit_path` (str): Path to DiT model
- `te_path` (str): Path to text encoder model
- `vae_path` (str): Path to VAE model
- `sampler_name` (str): Sampler plugin name
- `scheduler_name` (str): Scheduler plugin name
- `prompt_processor_name` (str): Prompt processor plugin name

**Returns:**
- List of PIL Images

### Qwen2TextEncoder

Text encoding class.

**Methods:**
- `__init__(gguf_path, device="cuda", dtype=torch.float16)`: Initialize encoder
- `encode(input_ids)`: Encode token IDs to embeddings

### FluxDiT

Diffusion transformer class.

**Methods:**
- `__init__(gguf_path, device="cuda", dtype=torch.bfloat16)`: Initialize model
- `forward(img_latent, txt_hidden, timestep, img_h, img_w)`: Forward pass

### load_vae()

Load VAE decoder.

**Parameters:**
- `safetensors_path` (str): Path to VAE weights
- `device` (str): Compute device
- `dtype` (torch.dtype): Data type

**Returns:**
- VAE model instance

## Performance Optimization

### Memory Management
```python
# Force garbage collection between generations
import gc
images = generate_image(prompts="...")
gc.collect()
torch.cuda.empty_cache()
```

### Mixed Precision
```python
# Use FP16 for faster inference (may reduce quality slightly)
images = generate_image(
    prompts="...",
    dtype=torch.float16
)
```

### Batch Processing Tips
```python
# Process in smaller batches to manage memory
batch_size = 2
all_images = []
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_seeds = seeds[i:i+batch_size]
    images = generate_image(
        prompts=batch_prompts,
        seeds=batch_seeds,
        # ... other params
    )
    all_images.extend(images)
```

## Error Handling

```python
try:
    images = generate_image(prompts="your prompt")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"CUDA/GPU error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

### Creative Writing Assistant
```python
def generate_scene_image(scene_description):
    return generate_image(
        prompts=scene_description,
        height=768,
        width=1024,
        num_steps=40,
        sampler_name="euler_a"
    )[0]

# Usage
image = generate_scene_image("A mystical forest with glowing mushrooms and floating lanterns")
image.save("scene.png")
```

### Style Transfer Pipeline
```python
def apply_art_style(image_prompt, style):
    styled_prompt = f"{image_prompt}, in the style of {style}"
    return generate_image(
        prompts=styled_prompt,
        num_steps=30,
        scheduler_name="karras"
    )[0]

styles = ["Van Gogh", "Picasso", "Monet"]
for style in styles:
    img = apply_art_style("a serene lake landscape", style)
    img.save(f"lake_{style.lower()}.png")
```

### Interactive Generation
```python
import gradio as gr

def generate_from_text(prompt, steps, seed):
    images = generate_image(
        prompts=prompt,
        num_steps=int(steps),
        seeds=int(seed)
    )
    return images[0]

# Gradio interface
gr.Interface(
    fn=generate_from_text,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(10, 100, value=25, label="Steps"),
        gr.Number(value=42, label="Seed")
    ],
    outputs=gr.Image()
).launch()
```

## Contributing

GGUF-Text-to-Image welcomes contributions! Areas for improvement:

- Additional sampler algorithms
- New noise schedules
- Prompt engineering techniques
- Performance optimizations
- Documentation improvements

Please submit pull requests to https://github.com/g023/gguf-text-to-image

## Changelog

### v1.0.0 (2026-02-11)
- Initial release
- GGUF quantization support
- Plugin system
- Batch processing
- Command-line interface</content>
<parameter name="filePath">/workspace/RELEASE/README.PYTHON.md