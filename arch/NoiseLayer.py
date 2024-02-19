from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as func

class NoiseLayer(nn.Module):
    """
    Noise Layer

    Adapted from Padilla-Zepeda E, Torres-Roman D, Mendez-Vazquez A. A Semantic Segmentation Framework for Hyperspectral Imagery Based on Tucker Decomposition and 3DCNN Tested with Simulated Noisy Scenarios[J]. Remote Sensing, 2023, 15(5): 1399.

    Add support for dark current
    """
    def __init__(self, SNR, alpha, dc=0 , bitdepth=8, seed=-1) -> None:
        super(NoiseLayer,self).__init__()
        self.SNR = SNR
        self.alpha = alpha
        self.dc = dc
        self.bitdepth = bitdepth
        self.add_noise = True if SNR < 100 else False
        self.quantization = True if bitdepth >0 else False
        self.seed = seed

    def forward(self, input):
        # input: [batch_size, channel, height, width]

        if self.seed > 0:
            seed_state = torch.random.get_rng_state()
            torch.manual_seed(self.seed)

        # calculate the power of the input signal

        if self.add_noise:

            input_shape = tuple(input.shape)

            input_power = torch.mean(torch.pow(input, 2))

            sigma = input_power * (10 ** (-self.SNR / 10))

            sigma_sd = sigma * self.alpha / (1+self.alpha)
            sigma_si = sigma / (1+self.alpha)

            RandomVariable_sd = torch.normal(mean=0., std=torch.sqrt(sigma_sd).item(), size=input_shape, device=input.device)

            dependent_noise = input * RandomVariable_sd

            independent_noise = torch.normal(mean=0., std=torch.sqrt(sigma_si).item(), size=input_shape, device=input.device)

            noise = dependent_noise + independent_noise
            noisy_data = input + noise

            # dark current
            if self.dc > 0:
                noisy_data = noisy_data + self.dc
        else:
            noisy_data = input

        if self.quantization:

            # quantization
            L = 2 ** self.bitdepth - 1
            noisy_data = torch.round(noisy_data * L) / L
            noisy_data = torch.clamp(noisy_data, min=0)
        else:
            noisy_data = noisy_data

        # reset seed state
        if self.seed > 0:
            torch.random.set_rng_state(seed_state)

        return noisy_data


class ClearNoiseLayer(nn.Module):
    def __init__(self, bitdepth=8) -> None:
        super(ClearNoiseLayer,self).__init__()
        self.bitdepth = bitdepth

    def forward(self, input):
        # input: [batch_size, channel, height, width]

        # quantization
        L = 2 ** self.bitdepth - 1
        noisy_data = torch.round(input * L) / L
        noisy_data = torch.clamp(noisy_data, 0, 1)

        return noisy_data
    
class NoiseLayer_Classic(nn.Module):
    def __init__(self, amp, bitdepth=8, seed=-1) -> None:
        super(NoiseLayer_Classic,self).__init__()
        self.amp = amp
        self.bitdepth = bitdepth
        self.seed = seed

    def forward(self, input):
        # input: [batch_size, channel, height, width]

        if self.seed > 0:
            seed_state = torch.random.get_rng_state()
            torch.manual_seed(self.seed)

        # add noise

        noisy_data = input + torch.randn_like(input) * self.amp * input

        # quantization
        L = 2 ** self.bitdepth - 1
        noisy_data = torch.round(noisy_data * L) / L
        noisy_data = torch.clamp(noisy_data, 0, 1)

        # reset seed state
        if self.seed > 0:
            torch.random.set_rng_state(seed_state)

        return noisy_data