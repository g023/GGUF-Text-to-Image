"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Flow matching scheduler with dynamic exponential time shift.
"""
import numpy as np
from plugins import PluginRegistry
from plugins.base import BaseScheduler


@PluginRegistry.register('scheduler', 'flow_match')
class FlowMatchScheduler(BaseScheduler):
    """Flow matching scheduler with exponential time shift.
    
    Matches the diffusers QwenImage pipeline schedule exactly.
    Uses dynamic mu based on image sequence length.
    """
    name = "flow_match"

    def __init__(self, base_seq_len=256, max_seq_len=4096,
                 base_shift=0.5, max_shift=1.15):
        self.base_seq_len = base_seq_len
        self.max_seq_len = max_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift

    def _calculate_shift(self, image_seq_len):
        m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len)
        b = self.base_shift - m * self.base_seq_len
        return image_seq_len * m + b

    def get_sigmas(self, num_steps, image_seq_len=1024):
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        mu = self._calculate_shift(image_seq_len)
        shifted = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
        return np.append(shifted, 0.0)

    def get_mu(self, image_seq_len):
        return self._calculate_shift(image_seq_len)
