from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.module import Module
from tmm_torch import TMM_predictor

class SWNet(nn.Sequential):
    """
    A neural network model that consists of multiple linear layers with leaky ReLU activation function.

    Args:
    - size (list): A list of integers representing the size of each layer in the network.
    - device (str): A string representing the device to be used for computation.

    Returns:
    - A neural network model with multiple linear layers and leaky ReLU activation function.
    """
    def __init__(self,size, device):
        super(SWNet, self).__init__()
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        for i in range(1, len(size) - 1):
            self.add_module('Linear' + str(i), nn.Linear(size[i], size[i + 1]))
            # self.SWNet.add_module('BatchNorm' + str(i), nn.BatchNorm1d(size[i+1]))
            # self.SWNet.add_module('DropOut' + str(i), nn.Dropout(p=0.2))
            self.add_module('LReLU' + str(i), nn.LeakyReLU(inplace=True))

        self.to(device)

class HybridNet(nn.Module):
    """
    A neural network model that combines a fixed neural network (fnet) with a switchable neural network (SWNet).
    The design parameters of the fnet are learned by the model.

    Args:
    - fnet_path (str): The file path to the pre-trained fnet model.
    - thick_min (float): The minimum thickness value for the design parameters.
    - thick_max (float): The maximum thickness value for the design parameters.
    - size (tuple): The input size of the SWNet.
    - device (str): The device to run the model on.
    - QEC (int or numpy.ndarray): The quantum error correction (QEC) value. Default is 1.

    Attributes:
    - fnet (nn.Module): The pre-trained fixed neural network.
    - tf_layer_num (int): The number of layers in the fnet.
    - DesignParams (nn.Parameter): The design parameters of the fnet.
    - QEC (int or torch.Tensor): The quantum error correction (QEC) value.
    - SWNet (nn.Module): The switchable neural network.
    """

    def __init__(self, fnet_path, thick_min, thick_max, size, device, QEC=1):
        super(HybridNet, self).__init__()

        # Load the pre-trained fnet model
        self.fnet = torch.load(fnet_path)
        self.fnet.to(device)
        self.fnet.eval()

        # Determine the number of layers in the fnet
        if isinstance(self.fnet, TMM_predictor):
            self.tf_layer_num = self.fnet.num_layers -2
        else:
            for p in self.fnet.parameters():
                p.requires_grad = False
            self.tf_layer_num = self.fnet.state_dict()['0.weight'].data.size(1)

        # Initialize the design parameters of the fnet
        self.DesignParams = nn.Parameter(
            (thick_max - thick_min) * torch.rand([size[1], self.tf_layer_num])*0+50 + thick_min, requires_grad=True)

        # Set the QEC value
        self.QEC = QEC
        if not isinstance(self.QEC, int):
            self.QEC = torch.from_numpy(self.QEC).float().to(device)
            self.QEC.requires_grad = False

        # Initialize the SWNet
        self.SWNet = SWNet(size,device)
        self.to(device)

    def forward(self, data_input):
        """
        Forward pass of the model.

        Args:
        - data_input (torch.Tensor): The input data.

        Returns:
        - The output of the SWNet.
        """
        sampled = func.linear(
            data_input, 
            self.fnet(self.DesignParams) * self.QEC,
            None
        )
        return self.SWNet(sampled)

    def show_design_params(self):
        """
        Returns the design parameters of the fnet.

        Returns:
        - The design parameters of the fnet.
        """
        return self.DesignParams
    
    def set_design_params(self,design_params:torch.Tensor):
        """
        Sets the design parameters of the fnet.

        Args:
        - design_params (torch.Tensor): The new design parameters.

        Raises:
        - AssertionError: If the size of the new design parameters does not match the size of the current design parameters.
        """
        try:
            assert design_params.size() == self.DesignParams.size()
        except AssertionError:
            print(f"design_params.size() = {design_params.size()}, self.DesignParams.size() = {self.DesignParams.size()}")
            raise AssertionError

        self.DesignParams.data = design_params.data
    
    def show_hw_weights(self):
        """
        Returns the hardware weights of the fnet.

        Returns:
        - The hardware weights of the fnet.
        """
        return self.fnet(self.DesignParams)

    def eval_fnet(self):
        """
        Sets the fnet to evaluation mode.

        Returns:
        - 0 (int): A dummy value.
        """
        self.fnet.eval()
        return 0

    def run_fnet(self, design_params_input):
        """
        Runs the fnet with the given design parameters.

        Args:
        - design_params_input (torch.Tensor): The design parameters to run the fnet with.

        Returns:
        - The output of the fnet.
        """
        return self.fnet(design_params_input)

    def run_swnet(self, data_input, hw_weights_input):
        """
        Runs the SWNet with the given input data and hardware weights.

        Args:
        - data_input (torch.Tensor): The input data.
        - hw_weights_input (torch.Tensor): The hardware weights to run the SWNet with.

        Raises:
        - AssertionError: If the size of the hardware weights does not match the size of the design parameters.

        Returns:
        - The output of the SWNet.
        """
        assert hw_weights_input.size(0) == self.DesignParams.size(0)
        return self.SWNet(func.linear(data_input, hw_weights_input, None))


class NoisyHybridNet(HybridNet):
    def __init__(self, fnet_path, thick_min, thick_max, size, noise_layer,device, QEC=1):
        super(NoisyHybridNet, self).__init__(fnet_path, thick_min, thick_max, size,device, QEC)
        
        self.noise_layer = noise_layer
        self.noise_layer.to(device)


    def forward(self, data_input):
        sampled = func.linear(data_input, self.fnet(self.DesignParams)*self.QEC, None)
        sampled = self.noise_layer(sampled)
        return self.SWNet(sampled)
    
    def run_swnet(self, data_input, hw_weights_input):
        sampled = self.noise_layer(func.linear(data_input, hw_weights_input, None))
        return self.SWNet(sampled)

class ND_HybridNet(NoisyHybridNet):
    def __init__(self, diff_row ,fnet_path, thick_min, thick_max, size, noise_layer,device,QEC=1):
        super(ND_HybridNet, self).__init__(fnet_path, thick_min, thick_max, size, noise_layer,device)
        self.diff_row = diff_row
        self.original_idx = torch.arange(size[1], device=device)
        self.diff_idx = torch.arange(size[1], device=device).reshape(self.diff_row, -1).roll(1, dims=1).reshape(-1)
        self.to(device)

    def forward(self, data_input):
        response = self.fnet(self.DesignParams)*self.QEC
        sampled = func.linear(data_input, response, None)
        sampled = self.noise_layer(sampled)
        diffed_sampled = sampled[:, self.diff_idx] - sampled[:, self.original_idx]
        return self.SWNet(diffed_sampled)
    
    def run_swnet(self, data_input, hw_weights_input):
        diffed_response = hw_weights_input[:, self.diff_idx] - hw_weights_input[:, self.original_idx]
        sampled = self.noise_layer(func.linear(data_input, diffed_response, None))
        return self.SWNet(sampled)


def MRAE(t1, t2):
    return torch.mean(torch.abs(t1 - t2) / torch.abs(t1))


MatchLossFcn = nn.MSELoss(reduction='mean')
# MatchLossFcn = MRAE




class HybnetLoss(nn.Module):
    def __init__(self):
        super(HybnetLoss, self).__init__()

    def forward(self, t1, t2, params, thick_min, thick_max, beta_range):
        """
        Calculates the loss for the HybridNet model.

        Args:
            t1 (torch.Tensor): The input tensor for the first image.
            t2 (torch.Tensor): The input tensor for the second image.
            params (torch.Tensor): The structure parameters for the model.
            thick_min (float): The minimum thickness value for the structure parameters.
            thick_max (float): The maximum thickness value for the structure parameters.
            beta_range (float): The regularization parameter for the structure parameter range.

        Returns:
            torch.Tensor: The total loss for the HybridNet model.
        """
        # MSE loss
        match_loss = MatchLossFcn(t1, t2)

        # Filter loss: square of the difference between total thickness 
        filter_loss = torch.var(torch.sum(params, dim=1))/1000
        filter_loss = 0

        # Structure parameter range regularization.
        # U-shaped function，U([param_min + delta, param_max - delta]) = 0, U(param_min) = U(param_max) = 1。
        delta = 0.01
        res = torch.max((params - thick_min - delta) / (-delta), (params - thick_max + delta) / delta)
        range_loss = torch.mean(torch.max(res, torch.zeros_like(res)))

        return match_loss + beta_range * range_loss + beta_range * filter_loss
    
class HybnetLoss_plus(HybnetLoss):
    def __init__(self):
        super(HybnetLoss_plus, self).__init__()
    
    def forward(self, *args, responses=None):
        original_loss = super(HybnetLoss_plus, self).forward(*args)

        rloss = 0
        # calculate the cosine similarity between the responses
        if not responses is None:
            TFNum = responses.size(0)
            for i in range(TFNum):
                for j in range(i+1, TFNum):
                    rloss += torch.cosine_similarity(responses[i], responses[j], dim=0)
            rloss /= TFNum * (TFNum-1) / 2

            rloss = rloss **2


        return original_loss + rloss






