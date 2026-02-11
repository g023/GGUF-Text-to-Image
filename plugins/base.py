"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Abstract base classes for pipeline plugins.
"""
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for diffusion samplers.
    
    A sampler defines how to update the sample at each denoising step
    given the model's velocity/noise prediction.
    """
    name: str = "base"

    @abstractmethod
    def step(self, model_output, sigma, sigma_next, sample):
        """Perform one denoising step.
        
        Args:
            model_output: Velocity prediction from DiT [B, S, D]
            sigma: Current noise level (float)
            sigma_next: Next noise level (float)
            sample: Current noisy sample [B, S, D]
        Returns:
            Updated sample tensor [B, S, D]
        """
        pass


class BaseScheduler(ABC):
    """Base class for noise schedulers.
    
    A scheduler defines the sequence of noise levels (sigmas) used
    during the denoising process.
    """
    name: str = "base"

    @abstractmethod
    def get_sigmas(self, num_steps, **kwargs):
        """Generate the noise schedule.
        
        Args:
            num_steps: Number of denoising steps
            **kwargs: Additional parameters (e.g., image_seq_len)
        Returns:
            numpy array of sigma values with terminal 0.0 appended
        """
        pass


class BasePromptProcessor(ABC):
    """Base class for prompt processors.
    
    A prompt processor handles formatting/templating of user prompts
    before tokenization and encoding.
    """
    name: str = "base"

    @abstractmethod
    def process(self, prompt, tokenizer):
        """Process a prompt into token IDs.
        
        Args:
            prompt: Raw user prompt string
            tokenizer: Tokenizer instance
        Returns:
            Tuple of (input_ids tensor [1, seq_len], drop_idx int)
            where drop_idx is the number of template prefix tokens to drop
        """
        pass
