import numpy as np
import torch

def shift_array(array, r_ratio, d_ratio):
    r_dir, d_dir = int(r_ratio//1), int(d_ratio//1)
    _array = np.roll(array, r_dir, axis=1)
    _array = np.roll(_array, d_dir, axis=0)
    r_ratio, d_ratio = r_ratio%1, d_ratio%1
    arr_r = np.roll(_array, 1, axis=1)
    arr_d = np.roll(_array, 1, axis=0)
    arr_rd = np.roll(arr_d, 1, axis=1)
    arr_shifted = (1-r_ratio)*(1-d_ratio)*_array + r_ratio*(1-d_ratio)*arr_r + (1-r_ratio)*d_ratio*arr_d + r_ratio * d_ratio * arr_rd
    return arr_shifted

def shift_array_torch(array, r_ratio, d_ratio, dims=(0,1)):
    r_dir, d_dir = int(r_ratio//1), int(d_ratio//1)
    _array = torch.roll(array, shifts=r_dir, dims=dims[1])
    _array = torch.roll(_array, shifts=d_dir, dims=dims[0])
    r_ratio, d_ratio = r_ratio%1, d_ratio%1
    arr_r = torch.roll(_array, shifts=1, dims=dims[0])
    arr_d = torch.roll(_array, shifts=1, dims=dims[1])
    arr_rd = torch.roll(arr_d, shifts=1, dims=dims[0])
    arr_shifted = (1-r_ratio)*(1-d_ratio)*_array + r_ratio*(1-d_ratio)*arr_r + (1-r_ratio)*d_ratio*arr_d + r_ratio * d_ratio * arr_rd
    return arr_shifted
