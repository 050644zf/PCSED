import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

default_backend = 'numpy'
device = None
try:
    import torch
    if torch.cuda.is_available():
        default_backend = 'torch'
        device = torch.device('cuda')
except:
    pass


def downsampling_matrix(original_wl, target_wl):
    """
    Generate downsampling matrix from original_wl to target_wl using convolution
    :param original_wl: original wavelength
    :param target_wl: target wavelength
    """
    conv_matrix = np.zeros((original_wl.shape[0], original_wl.shape[0]))
    original_step = original_wl[1] - original_wl[0]
    target_step = target_wl[1] - target_wl[0]
    kernel_size = int(np.ceil(target_step / original_step)) + 1
    kernel = np.ones(kernel_size)
    kernel = kernel / np.sum(kernel)

    for i in range(original_wl.shape[0]):
        conv_matrix[:, i] = np.convolve(np.eye(original_wl.shape[0])[i], kernel, mode='same')

    # boundary compensation
    for i in range(int(np.floor(kernel_size / 2))):
        conv_matrix[0, i] += max(1 - conv_matrix[:, i].sum(), 0)
        conv_matrix[-1, -i - 1] += max(1 - conv_matrix[:, -i - 1].sum(), 0)

    # remove the redundant bands
    min_index = np.argmin(np.abs(original_wl - target_wl[0]))
    max_index = np.argmin(np.abs(original_wl - target_wl[-1]))

    return conv_matrix[:, min_index:max_index + 1:kernel_size - 1]

# 采样
def apply_sampling_matrix(img:np.ndarray, sampling_matrix:np.ndarray, backend='numpy')->np.ndarray:
    """
    Apply sampling matrix to image in spectral dimension
    :param img: input image, [h,w,n_in] or [n_px,n_in]
    :param sampling_matrix: sampling matrix [n_in, n_out]
    :return: sampled image
    """
    assert img.shape[-1] == sampling_matrix.shape[0], "image and sampling matrix should have same number of bands"
    if backend == 'numpy':
        if len(img.shape) == 2:
            return np.dot(img, sampling_matrix)
        return np.einsum('ijk,kl->ijl', img, sampling_matrix)
    elif backend == 'torch':
        img = torch.from_numpy(img).float().to(device)
        sampling_matrix = torch.from_numpy(sampling_matrix).float().to(device)
        if len(img.shape) == 2:
            return torch.matmul(img, sampling_matrix).cpu().numpy()
        return torch.matmul(img, sampling_matrix).cpu().numpy()

if __name__ == '__main__':
    mat_file_path = r'D:\learn\PCSED\data\SC035HGSI.mat'
    mat_data = loadmat(mat_file_path)
    down_mat = downsampling_matrix(np.arange(400, 1000.5, 0.5), np.arange(400, 1005, 5))
    data = mat_data['data']
    random_array = apply_sampling_matrix(data, down_mat)

    savemat('SC035HGSI_121.mat', {'data': random_array})

