"""
This script trains a hybrid neural network (HybNet) for the purpose of predicting the optical response of a thin film. 
The HybNet consists of a spectral weighting network (SWNet) and a transfer function network (TFNet). 
The SWNet is a neural network that takes in a spectrum and outputs a set of weights that are used to weight the transfer functions in the TFNet. 
The TFNet is a set of transfer functions that are used to predict the optical response of the thin film. 
The script loads training and testing data, and uses the HybNet to predict the optical response of the testing data. 
The script also logs the training process and saves the trained HybNet.
"""

# Import necessary libraries
import arch.HybridNet as HybridNet
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os
import json
import shutil
from pathlib import Path


from tmm_torch import TMM_predictor

# Set working directory to the directory of this script
os.chdir(Path(__file__).parent)

from load_config import *
from load_ADMM_data import *
from arch.ADMM_net import ADMM_net
from metric import torch_psnr, torch_ssim

train_set = LoadTraining(admm_config['TrainDataPath'])
test_data = LoadTest(admm_config['TestDataPath'])

# 加载光源
light_mat_path = admm_config['light']
light_mat = sio.loadmat(light_mat_path)
light_data = light_mat['data']

noise_level = noise_config['amp']

# Set size of HybNet and create HybNet object
hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]
hybnet = HybridNet.ADMM_HybridNet(fnet_path, params_min, params_max, hybnet_size, device_train, QEC=QEC)

checkpoint = torch.load(args.pretained)
hybnet.ADMM_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
hybnet.ADMM_net.to(device_train)
hybnet.ADMM_net.eval()

# Override filters if specified in configuration
if config.get("override_filters"):
    try:
        design_params = scio.loadmat(Path(config.get("override_filters")))["params"]
    except:
        design_params = scio.loadmat(Path(config.get("override_filters"))/"TrainedParams.mat")["Params"]
    hybnet.set_design_params(torch.tensor(design_params, device=device_train, dtype=dtype))
    hybnet.DesignParams.requires_grad = False
    hybnet.eval_fnet()

# Set loss function and optimizer
LossFcn = HybridNet.HybnetLoss_plus()
# optimizer_net = torch.optim.Adam(filter(lambda p: p.requires_grad, hybnet.SWNet.parameters()), lr=lr)
# scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=lr_decay_step, gamma=lr_decay_gamma) 
optimizer_params = torch.optim.Adam(filter(lambda p: p.requires_grad, [hybnet.DesignParams]), lr=lr*config.get("params_lr_coef",1))
scheduler_params = torch.optim.lr_scheduler.StepLR(optimizer_params, step_size=lr_decay_step, gamma=lr_decay_gamma)

# Initialize variables for logging and training
loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))
log_file = open(path / 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
params_history = [hybnet.show_design_params().detach().cpu().numpy()]

# 新增修改
n_filter=config['TFNum']
TFNum = fnet_config['TFNum']
image = torch.ones(BatchSize, 128, 128, 121)
params = torch.nn.Parameter(torch.rand(n_filter, TFNum) * (params_max - params_min)*0.1 + params_min)
params = params.cuda()


BatchSize = admm_config['batch_size']


def test(model, add_noise = False):
    test_gt = test_data.cuda().float().contiguous()
    model_out_list=[]
    test_gt_list = []
    for j in [0, 6, 7]:
        light = light_data[j, :]
        light = np.tile(light.reshape((1, 1, 121)), (test_gt.shape[0], 128, 128, 1))
        light = torch.from_numpy(light).cuda().float()
        light = light.permute(0, 3, 1, 2).contiguous()

        test_gt = test_gt * light
        input_meas_my = torch.sum(Phi * test_gt, 1)
        if add_noise:
            # 保存当前随机种子状态
            torch_state = torch.get_rng_state()
            torch.manual_seed(42)
            input_meas_my = input_meas_my + torch.randn_like(input_meas_my) * input_meas_my * 0.05
            # 恢复默认随机种子状态
            torch.set_rng_state(torch_state)

        with torch.no_grad():
            model_out = model(input_meas_my)
        model_out_list.append(model_out)
        test_gt_list.append(test_gt)
    model_out = torch.cat(model_out_list, 0)
    test_gt = torch.cat(test_gt_list, 0)
    return model_out, test_gt
    
def compute_metrics(model_out, test_gt):
    psnr_list, ssim_list = [], []
    for i in range(model_out.shape[0]):
        psnr_val = torch_psnr(model_out[i], test_gt[i])
        ssim_val = torch_ssim(model_out[i], test_gt[i])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    return np.mean(psnr_list), np.mean(ssim_list)

def log(text, file=log_file):
    print(text)
    print(text, file=file)

# Train Params
for epoch in range(EpochNum):
    # TODO: 改训练代码
    # Shuffle training data
    gt_batch = shuffle_crop(train_set, BatchSize, crop_size=128)

    # TODO: 这里可能要改成DataLoader
    for i in range(0, TrainingDataSize // BatchSize):
        # Get batch of training data
        gt_batch = shuffle_crop(train_set, BatchSize, crop_size=128)
        # 自动求导
        gt = Variable(gt_batch).cuda().float()

        # 更新Phi
        noised_params = params + (torch.rand_like(params) * 2 - 1) * thickness_error
        # noised_params = torch.rand(n_filter, TFNum) * (params_max - params_min) + params_min
        # noised_params = params
        responses = hybnet.fnet(noised_params)
        Phi_data_tensor = responses

        Phi = Phi_data_tensor


        # TODO: 改数据
        out_list, gt_list = [], []
        for j in [0,6,7]:
            light = light_data[j]
            light = np.tile(light.reshape((1, 1, 121)), (gt.shape[0], 128, 128, 1))
            light = torch.from_numpy(light).cuda().float()
            light = light.permute(0,3,1,2).contiguous()
            gt = gt * light
            gt_list.append(gt)
            input_meas_my = torch.sum(Phi *gt, 1)
            input_meas_my = input_meas_my + torch.randn_like(input_meas_my) * input_meas_my * noise_level
            # Forward pass through HybNet
            model_out = hybnet(input_meas_my)
            out_list.append(model_out)

            DesignParams = hybnet.show_design_params()
            responses = hybnet.show_hw_weights()

        model_out = torch.cat(out_list, 0)
        gt = torch.cat(gt_list, 0)            
        # Calculate loss and backpropagate
        loss = LossFcn(model_out, gt, DesignParams, params_min.to(device_train), params_max.to(device_train), beta_range,responses=responses)
        optimizer_params.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_params.step()
    scheduler_params.step()
    if epoch % TestInterval == 0:
        # Evaluate HybNet on testing data
        hybnet.to(device_test)
        # hybnet.eval_fnet()

        DesignParams = hybnet.show_design_params()
        responses = hybnet.show_hw_weights()
        if config.get("History"):
            params_history.append(DesignParams.detach().cpu().numpy())
            scio.savemat(path / "params_history.mat", {"params_history": params_history})


        Out_test_pred, Specs_test = test(hybnet, add_noise=False)

        loss_train[epoch // TestInterval] = loss.data
        # TODO: 改Test loss的输入
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data

        psnr_list, ssim_list = compute_metrics(Out_test_pred, Specs_test)
        psnr_mean = np.mean(psnr_list)
        ssim_mean = np.mean(ssim_list)

        test_result_folder = path / "test_result" / f"epoch_{epoch}_{psnr_mean:.2f}_{ssim_mean:.4f}"

        test_result_folder.mkdir(parents=True, exist_ok=True)

        torch.save(hybnet, test_result_folder / "hybnet.pkl")
        scio.savemat(test_result_folder / 'params.mat', {"params": DesignParams.detach().cpu().numpy(), 'response': responses.detach().cpu().numpy()})


        # TODO: 改PSNR, SSIM等输出
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)

        log(f'Epoch {epoch}/{EpochNum} training loss: {loss.data:.4f}, testing loss: {loss_t.data:.4f}, PSNR: {psnr_mean:.2f}, SSIM: {ssim_mean:.4f}, time remain: {time_remain:.0f}s')

        # test under noise
        Out_test_pred, Specs_test = test(hybnet, add_noise=True)
        psnr_list, ssim_list = compute_metrics(Out_test_pred, Specs_test)
        psnr_mean = np.mean(psnr_list)
        ssim_mean = np.mean(ssim_list)
        log(f'Epoch {epoch}/{EpochNum} training loss: {loss.data:.4f}, testing loss @ {noise_level} noise: {loss_t.data:.4f}, PSNR: {psnr_mean:.2f}, SSIM: {ssim_mean:.4f}, time remain: {time_remain:.0f}s')

time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

hybnet.eval()
hybnet.eval_fnet()
torch.save(hybnet, path / 'hybnet.pkl')
hybnet.to(device_test)

HWweights = hybnet.show_hw_weights()
TargetCurves = HWweights.double().detach().cpu().numpy()
scio.savemat(path / 'TargetCurves.mat', mdict={'TargetCurves': TargetCurves})

DesignParams = hybnet.show_design_params()
print(DesignParams[0, :])
TargetCurves_FMN = hybnet.run_fnet(DesignParams).double().detach().cpu().numpy()
scio.savemat(path / 'TargetCurves_FMN.mat', mdict={'TargetCurves_FMN': TargetCurves_FMN})
Params = DesignParams.double().detach().cpu().numpy()
scio.savemat(path / 'TrainedParams.mat', mdict={'Params': Params})
