#!/bin/bash

# Copyright (c) 2026, g023 (https://github.com/g023)
# BSD 3-Clause License. See README.PYTHON.md for details.

# GGUF-Text-to-Image Installation Script
# Installs all required dependencies for the GGUF-Text-to-Image text-to-image generator

set -e  # Exit on any error

echo "========================================"
echo "  GGUF-Text-to-Image Dependency Installer"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed"
    echo "Please install pip and try again"
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
fi

echo "✓ Pip found: $($PIP_CMD --version)"
echo

echo "Installing PyTorch with CUDA support..."
echo "This may take a few minutes..."
$PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo
echo "Installing core dependencies..."
$PIP_CMD install transformers diffusers safetensors pillow numpy gguf

echo
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo
echo "GGUF-Text-to-Image is ready to use."
echo "Run: python3 generate.py --help"
echo
echo "For more information, see README.md and README.PYTHON.md"