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

# TODO: 加载数据 需要添加路径
train_set = LoadTraining(train_data_path)
test_data = LoadTest(test_data_path)


# Set size of HybNet and create HybNet object
hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]
hybnet = HybridNet.ADMM_HybridNet(fnet_path, params_min, params_max, hybnet_size, device_train, QEC=QEC)

hybnet.ADMM_net = torch.load(args.pretained)
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
n_filter=9
image = torch.ones(BatchSize, 128, 128, 121)
params = torch.nn.Parameter(torch.rand(n_filter, TFNum) * (params_max - params_min)*0.1 + params_min)
params = params.cuda()
# Train Params
for epoch in range(EpochNum):
    # TODO: 改训练代码
    # Shuffle training data
    # Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]

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
        responses = fnet(noised_params)
        Phi_data_tensor = responses

        Phi = Phi_model.get_Phi(image, Phi_data_tensor)
        Phi = Phi.permute(0, 3, 1, 2).cuda()
        Phi = Phi.contiguous()
        Phi_s = torch.sum(Phi ** 2, 1)
        Phi_s[Phi_s == 0] = 1
        input_mask_train = (Phi, Phi_s)


        # TODO: 改数据
        Specs_batch = 
        # Forward pass through HybNet
        Output_pred = hybnet(Specs_batch)
        DesignParams = hybnet.show_design_params()
        responses = hybnet.show_hw_weights()
        # Calculate loss and backpropagate
        loss = LossFcn(Specs_batch, Output_pred, DesignParams, params_min.to(device_train), params_max.to(device_train), beta_range,responses=responses)
        optimizer_params.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_params.step()
    scheduler_params.step()
    if epoch % TestInterval == 0:
        # Evaluate HybNet on testing data
        hybnet.to(device_test)
        with torch.no_grad():
            # TODO: 加测试代码，算PSNR等
            pass


        hybnet.eval_fnet()

        DesignParams = hybnet.show_design_params()
        if config.get("History"):
            params_history.append(DesignParams.detach().cpu().numpy())
            scio.savemat(path / "params_history.mat", {"params_history": params_history})

        loss_train[epoch // TestInterval] = loss.data
        # TODO: 改Test loss的输入
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data


        # TODO: 改PSNR, SSIM等输出
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler_params.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler_params.get_lr()[0], file=log_file)
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
