"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Euler Ancestral sampler - adds stochastic noise at each step.
"""
import torch
from plugins import PluginRegistry
from plugins.base import BaseSampler


@PluginRegistry.register('sampler', 'euler_a')
class EulerAncestralSampler(BaseSampler):
    """Euler method with ancestral (stochastic) sampling.
    
    Adds controlled noise at each step, which can improve diversity
    and detail at the cost of some determinism.
    """
    name = "euler_a"

    def __init__(self, eta=1.0):
        self.eta = eta

    def step(self, model_output, sigma, sigma_next, sample):
        if sigma_next == 0 or self.eta == 0:
            # Final step or no noise: pure Euler
            dt = sigma_next - sigma
            return sample + dt * model_output

        # Compute noise injection magnitude
        sigma_up = min(
            sigma_next,
            self.eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5
        )
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5

        # Euler step to sigma_down
        dt = sigma_down - sigma
        sample = sample + dt * model_output

        # Add noise scaled by sigma_up
        noise = torch.randn_like(sample)
        sample = sample + sigma_up * noise

        return sample
