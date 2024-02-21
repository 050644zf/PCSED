import torch
from torch import nn
import torch.nn.functional as func

class Mosiac_Layer(nn.Module):
    def __init__(self, shape:list, pattern=None) -> None:
        """
        Apply the response curve on the input image based on the mosiac pattern.
        :param shape: the shape of the mosiac pattern, shape: [n_h, n_w]
        :param pattern: the mosiac pattern, contain the channel index, shape: [n_h, n_w]

        For example, an RGGB pattern is:
        pattern = torch.tensor([[0, 1],
                                [1, 2]])
        """
        super().__init__()
        self.n_channel = shape[0] * shape[1]
        self.shape = shape

        if pattern is None:
            self.pattern = torch.arange(self.n_channel)
            self.pattern = self.pattern.reshape(shape[0], shape[1])
        else:
            self.pattern = pattern


    def forward(self, input, Phi):
        """
        :param input: input hyperspectral image, shape: [batch_size, h, w, n_lambda]
        :param Phi: the response curve, shape: [n_channel, n_lambda]
        :return: the mosiac image, shape: [batch_size, h, w]
        """

        batch_size, h, w, n_lambda = input.shape

        mosiac_image = torch.zeros([batch_size, h, w], device=input.device, dtype=input.dtype)

        for c in range(self.n_channel):
            i = c // self.shape[1]
            j = c % self.shape[1]
            # mosiac_image += input[:, i::self.shape[0], j::self.shape[1], :]  Phi[self.pattern[i][j], :]
            mosiac_image[:,i::self.shape[0], j::self.shape[1]] += func.linear(input[:, i::self.shape[0], j::self.shape[1], :], Phi[self.pattern[i][j]])

        return mosiac_image
    
    def get_pattern(self):
        return self.pattern

    def multi_channel(self, input, Phi):
        """
        :param input: input hyperspectral image, shape: [batch_size, h, w, n_lambda]
        :param Phi: the response curve, shape: [n_channel, n_lambda]
        :return: the multichannel image, shape: [batch_size, h, w, n_channel]
        """

        batch_size, h, w, n_lambda = input.shape

        n_channel = Phi.shape[0]

        mosiac_image = torch.zeros([batch_size, h, w, n_channel], device=input.device, dtype=input.dtype)

        for c in range(n_channel):
            mosiac_image[:,:,:,c] = func.linear(input , Phi[c, :])

        return mosiac_image
    
    def get_Phi(self,input, Phi):
        """
        :param input: input hyperspectral image, shape: [batch_size, h, w, n_lambda]
        :param Phi: the response curve, shape: [n_channel, n_lambda]
        :return: the multichannel Phi, shape: [batch_size, h, w, n_lambda]
        """

        batch_size, h, w, n_lambda = input.shape

        n_channel = Phi.shape[0]

        mosiac_Phi = torch.zeros([batch_size, h, w, n_lambda], device=input.device, dtype=input.dtype)

        for c in range(self.n_channel):
            i = c // self.shape[1]
            j = c % self.shape[1]
            # mosiac_image += input[:, i::self.shape[0], j::self.shape[1], :]  Phi[self.pattern[i][j], :]
            mosiac_Phi[:,i::self.shape[0], j::self.shape[1],:] = Phi[self.pattern[i][j]]

        return mosiac_Phi

class RGGB_Layer(Mosiac_Layer):
    def __init__(self) -> None:
        super().__init__(shape=[2,2], pattern=[[0, 1],[1, 2]])

    

class SFA9(Mosiac_Layer):
    def __init__(self) -> None:
        super().__init__(shape=[3,3])

class SFA9_Binary(Mosiac_Layer):
    def __init__(self) -> None:
        super().__init__(
            shape=[4,4],
            pattern=[
                [3,0,4,8],
                [1,2,7,6],
                [4,8,5,0],
                [7,6,1,2]
            ]
        )

