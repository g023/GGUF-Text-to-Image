# GGUF-Text-to-Image: GGUF-Powered Text-to-Image Generation

**GGUF-Text-to-Image** is a high-performance, standalone Python library for text-to-image generation using GGUF-quantized diffusion models. Built for efficiency on consumer GPUs, it provides both command-line and programmatic interfaces for generating images from text prompts. Currently ONLY tested working with **Qwen-Image** GGUF models.

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
  - `Qwen-Rapid-NSW-v23_Q3_K.gguf` 
  - 'qwen_image_vae.safetensors'
  - `Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf`

# Qwen-Image GGUF Text-to-Image Generator

A standalone Python script for generating high-quality images from text prompts using Qwen-Image diffusion models. This implementation uses GGUF quantized models to enable efficient inference on consumer GPUs with 12GB VRAM, supporting batch processing and a modular plugin system for customization.

## Overview

This pipeline implements a complete text-to-image generation workflow based on the Qwen-Image architecture:

1. **Text Encoding**: Converts text prompts into rich embeddings using a Qwen2.5 language model.
2. **Latent Diffusion**: Uses a Diffusion Transformer (DiT) to denoise random noise into structured latent representations.
3. **Image Decoding**: Transforms latents back to pixel space using a Variational Autoencoder (VAE).

The system is designed for efficiency, processing images sequentially on GPU while maintaining memory constraints suitable for 12GB GPUs.

## Key Features

### 🚀 Performance Optimized
- **GGUF Quantization**: Models are quantized to Q3_K/Q4_K formats, reducing memory usage by 60-75% compared to FP16.
- **Sequential Processing**: Components load and process one at a time to fit within GPU memory limits.
- **Batch Support**: Generate multiple images in parallel from different prompts.
- **CUDA Acceleration**: Full GPU utilization with optimized tensor operations.

### 🎨 Generation Capabilities
- **High Resolution**: Supports up to 1024x1024 images (tested on 512x512).
- **Flexible Dimensions**: Custom height/width with aspect ratio preservation.
- **Quality Control**: Adjustable denoising steps (20-100 recommended).
- **Reproducibility**: Seed-based generation for consistent results.
- **Batch Processing**: Generate multiple images simultaneously.

### 🔧 Modular Architecture
- **Plugin System**: Extensible samplers, schedulers, and prompt processors.
- **Multiple Samplers**: Euler, Euler Ancestral for different generation styles.
- **Scheduler Options**: Flow Matching, Linear, Karras noise schedules.
- **Prompt Processing**: Qwen Chat formatting and raw text support.

### 📦 Self-Contained
- **No Dependencies**: Pure Python with minimal external libraries.
- **Standalone**: No ComfyUI or web UI required.
- **Easy Distribution**: Single folder with all necessary files.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (12GB+ VRAM recommended)
- Linux/Windows/macOS

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers safetensors pillow numpy gguf
```

### Setup
1. Clone or download the RELEASE folder
2. Ensure all files are present:
   ```
   RELEASE/
   ├── generate.py
   ├── text_encoder.py
   ├── dit.py
   ├── vae.py
   ├── gguf_utils.py
   ├── plugins/
   │   ├── __init__.py
   │   ├── base.py
   │   ├── samplers/
   │   ├── schedulers/
   │   └── prompt_processors/
   └── GGUF/
       ├── Qwen-Rapid-NSW-v23_Q3_K.gguf
       ├── Qwen-Rapid-NSW-v23_Q4_K.gguf
       ├── Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf
       └── qwen_image_vae.safetensors
   ```

## Usage

### Basic Usage

Generate a single image:
```bash
python3 generate.py --prompt "a beautiful sunset over mountains"
```

### Advanced Options

#### Single Image Generation
```bash
python3 generate.py \
  --prompt "a cute fluffy orange tabby cat sitting on a windowsill, soft lighting" \
  --height 512 \
  --width 512 \
  --steps 25 \
  --seed 123 \
  --output output_cat.png
```

#### Batch Generation
```bash
python3 generate.py \
  --prompts "sunset over mountains" "city at night" "forest landscape" \
  --height 512 \
  --width 512 \
  --steps 50 \
  --seeds 42 43 44 \
  --output batch_output.png
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | None | Single text prompt (mutually exclusive with --prompts) |
| `--prompts` | None | Multiple prompts for batch generation |
| `--height` | 512 | Output image height (must be multiple of 8) |
| `--width` | 512 | Output image width (must be multiple of 8) |
| `--steps` | 50 | Number of denoising steps (20-100 recommended) |
| `--seed` | None | Random seed for single image |
| `--seeds` | None | Random seeds for batch (one per prompt) |
| `--output` | output.png | Output filename (auto-appends _{i} for batches) |
| `--sampler` | euler | Sampler plugin: euler, euler_a |
| `--scheduler` | flow_match | Scheduler plugin: flow_match, linear, karras |
| `--prompt-processor` | qwen_chat | Prompt processor: qwen_chat, raw |

### Examples

#### Landscape Generation
```bash
python3 generate.py \
  --prompt "majestic mountain landscape with lake, dramatic lighting, photorealistic" \
  --height 768 \
  --width 1024 \
  --steps 30 \
  --seed 999 \
  --output mountain_landscape.png
```

#### Portrait Generation
```bash
python3 generate.py \
  --prompt "professional portrait of a woman, studio lighting, 85mm lens, sharp focus" \
  --height 512 \
  --width 512 \
  --steps 25 \
  --sampler euler_a \
  --scheduler karras \
  --output portrait.png
```

#### Batch Creative Prompts
```bash
python3 generate.py \
  --prompts \
    "cyberpunk city street at night, neon lights, rain, detailed" \
    "steampunk airship flying over Victorian London, dramatic clouds" \
    "minimalist geometric abstract art, bold colors, clean lines" \
  --height 512 \
  --width 512 \
  --steps 40 \
  --seeds 1001 1002 1003 \
  --output creative_batch.png
```

#### High Quality Mode
```bash
python3 generate.py \
  --prompt "hyper-realistic photograph of a rose, dew drops, macro lens, 8k resolution" \
  --height 512 \
  --width 512 \
  --steps 100 \
  --seed 42 \
  --output high_quality_rose.png
```

## Architecture & Algorithms

### Text Encoding Pipeline

The text encoder uses a Qwen2.5-7B model quantized to Q4_K_M format:

1. **Tokenization**: Input text is tokenized using the Qwen2.5 tokenizer
2. **Embedding**: Tokens are converted to 3584-dimensional embeddings
3. **Transformer Processing**: 28 layers of attention and MLP blocks
4. **RMS Normalization**: Stable normalization for better training stability
5. **RoPE Embeddings**: Rotary Position Embeddings for sequence understanding

**Key Features**:
- Processes up to 256 tokens per prompt
- Maintains contextual understanding across long prompts
- Optimized for creative and descriptive text

### Diffusion Transformer (DiT)

The core denoising model uses a 60-layer QwenImage DiT:

#### Architecture
- **Hidden Size**: 3072 dimensions
- **Attention Heads**: 24 heads (128 dim each)
- **Patch Size**: 2x2 patches (64 dimensions per patch)
- **Double-Stream Design**: Separate processing for text and image tokens

#### Key Mechanisms

**Patchification**:
- Images are divided into 2x2 patches
- Each patch becomes a 64-dimensional vector
- Spatial layout: H×W patches arranged in sequence

**RoPE Embeddings**:
- Complex rotary embeddings for spatial awareness
- Separate frequencies for height, width, and frame dimensions
- Enables understanding of 2D image structure

**Modulation**:
- Time embeddings modulate attention and MLP layers
- AdaLayerNorm-Continuous for adaptive normalization
- Scale and shift parameters control per-layer behavior

**Joint Attention**:
- Text and image tokens attend to each other
- Enables cross-modal understanding
- Text conditions image generation

### Denoising Process

#### Flow Matching
The default scheduler uses Flow Matching:

1. **Noise Schedule**: Smooth interpolation from noise to data
2. **Velocity Prediction**: Model predicts velocity field
3. **ODE Integration**: Euler method solves the flow ODE

**Mathematical Formulation**:
```
dx/dt = v(x, t)
x(0) = noise, x(1) = data
```

#### Samplers
- **Euler**: Simple first-order ODE solver
- **Euler Ancestral**: Adds stochasticity for variety

### Variational Autoencoder (VAE)

The decoder reconstructs images from latents:

#### Architecture
- **Latent Dimensions**: 16 channels, compressed 8x
- **Decoder**: 4 upsampling blocks with attention
- **Normalization**: Custom gamma normalization
- **Output**: RGB images in [-1, 1] range

#### Post-Processing
- Denormalization using learned mean/std
- Clamp to valid range
- Convert to uint8 for saving

### Plugin System

#### Samplers
Implement different ODE solvers:

- **Euler**: `dx = dt * v(x,t)`
- **Euler Ancestral**: Adds noise for diversity

#### Schedulers
Control noise level progression:

- **Flow Matching**: Smooth velocity-based schedule
- **Linear**: Simple linear noise decay
- **Karras**: Optimized for perceptual quality

#### Prompt Processors
Format prompts for the model:

- **Qwen Chat**: Conversational formatting
- **Raw**: Direct text input

## Performance Characteristics

### Memory Usage
- **Peak VRAM**: ~10-11GB for 512x512 generation
- **Sequential Loading**: Components loaded/unloaded as needed
- **Batch Efficiency**: Shared computation across prompts

### Generation Times
- **512x512, 25 steps**: ~45 seconds
- **512x512, 50 steps**: ~90 seconds
- **Batch of 3**: ~2.5x single image time

### Quality vs Speed Trade-offs
- **Steps**: 20-30 for fast, 50-100 for high quality
- **Resolution**: Higher resolution increases computation quadratically
- **Quantization**: Q3_K faster but slightly lower quality than Q4_K

## Model Files

### DiT Models
- **Qwen-Rapid-NSW-v23_Q3_K.gguf**: Fast, 3-bit quantization
- **Qwen-Rapid-NSW-v23_Q4_K.gguf**: Balanced quality/speed

### Text Encoder
- **Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf**: Optimized for vision-language tasks

### VAE
- **qwen_image_vae.safetensors**: Autoencoder for latent compression

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce resolution or steps
- Use Q3_K model instead of Q4_K
- Process one image at a time

**Slow Generation**:
- Use fewer steps (25-30)
- Switch to Q3_K quantization
- Ensure GPU is not shared with display

**Poor Quality**:
- Increase steps to 50+
- Use more descriptive prompts
- Try different seeds

**Installation Issues**:
- Ensure CUDA 12.1+ is installed
- Check PyTorch CUDA compatibility
- Install gguf: `pip install gguf`

### Debug Mode
Run with verbose output:
```bash
python3 generate.py --prompt "test" 2>&1
```

## Contributing

The plugin system allows easy extension:

1. Add new samplers in `plugins/samplers/`
2. Implement custom schedulers in `plugins/schedulers/`
3. Create prompt processors in `plugins/prompt_processors/`

Each plugin inherits from base classes in `plugins/base.py`.

## License

This implementation is based on Qwen-Image and uses GGUF quantization adapted from ComfyUI-GGUF (Apache-2.0).

## Acknowledgments

- Qwen-Image team for the original model architecture
- City96 for GGUF quantization utilities
- Hugging Face for transformers and diffusers libraries

## License

Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.</content>
