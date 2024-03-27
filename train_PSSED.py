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

# copy this file to the result folder
shutil.copy(Path(__file__).resolve(), path / Path(__file__).name)

fixed_chs = config.get("fixed_chs", None)
n_FIXED = 0
if not fixed_chs is None:
    fixed_chs = scio.loadmat(fixed_chs)["data"]
    n_FIXED = fixed_chs.shape[0]
    fixed_chs = torch.from_numpy(fixed_chs).float().to(device_train)
    


# Set size of HybNet and create HybNet object
hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]


hybnet = HybridNet.HybridNet(fnet_path, params_min, params_max, hybnet_size, device_train, QEC=QEC, fixed_chs=fixed_chs)



# Override filters if specified in configuration
if config.get("override_filters"):
    if config.get("override_filters").endswith(".mat"):
        design_params = scio.loadmat(Path(config.get("override_filters")))["Params"]
    else:
        design_params = scio.loadmat(Path(config.get("override_filters"))/"TrainedParams.mat")["Params"]
    hybnet.set_design_params(torch.tensor(design_params, device=device_train, dtype=dtype))
    hybnet.DesignParams.requires_grad = False
    hybnet.eval_fnet()

# Set loss function and optimizer
LossFcn = HybridNet.HybnetLoss_plus()
optimizer_net = torch.optim.Adam(filter(lambda p: p.requires_grad, hybnet.SWNet.parameters()), lr=lr)
# scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=lr_decay_step, gamma=lr_decay_gamma)
scheduler_net = HybridNet.Scheduler_net(optimizer_net, lr_decay_step, lr_decay_gamma, begin_epoch=50)
optimizer_params = torch.optim.Adam(filter(lambda p: p.requires_grad, [hybnet.DesignParams]), lr=lr*config.get("params_lr_coef",1))
scheduler_params = torch.optim.lr_scheduler.StepLR(optimizer_params, step_size=30, gamma=lr_decay_gamma)
# scheduler_params = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_params, T_max=EpochNum, eta_min=1e-2, verbose=True)

# Initialize variables for logging and training
loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))
log_file = open(path / 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
params_history = [hybnet.show_design_params().detach().cpu().numpy()]

# Train HybNet
for epoch in range(EpochNum):
    # Shuffle training data
    Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        # Get batch of training data
        Specs_batch = Specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        # Forward pass through HybNet
        Output_pred = hybnet(Specs_batch)
        DesignParams = hybnet.show_design_params()
        responses = hybnet.show_hw_weights()
        # Calculate loss and backpropagate
        loss = LossFcn(Specs_batch, Output_pred, DesignParams, params_min.to(device_train), params_max.to(device_train), beta_range, total_thickness_range,responses=responses)
        optimizer_net.zero_grad(),optimizer_params.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_net.step(),optimizer_params.step()
    scheduler_net.step(), scheduler_params.step()
    if epoch % TestInterval == 0:
        # Evaluate HybNet on testing data
        hybnet.to(device_test)
        hybnet.eval()
        Out_test_pred = hybnet(Specs_test)
        hybnet.to(device_train)
        hybnet.train()
        hybnet.eval_fnet()

        DesignParams = hybnet.show_design_params()
        if config.get("History"):
            params_history.append(DesignParams.detach().cpu().numpy())
            scio.savemat(path / "params_history.mat", {"params_history": params_history})

        loss_train[epoch // TestInterval] = loss.data
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler_net.get_lr()[0], '| params learn rate: %.8f' % scheduler_params.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler_net.get_lr()[0],  '| params learn rate: %.8f' % scheduler_params.get_lr()[0], file=log_file)
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

# plt.figure()
# for i in range(TFNum):
#     plt.subplot(math.ceil(math.sqrt(TFNum)), math.ceil(math.sqrt(TFNum)), i + 1)
#     plt.plot(WL, TargetCurves[i, :], WL, TargetCurves_FMN[i, :])
#     plt.ylim(0, 1)
# plt.savefig(path / 'ROFcurves')
# plt.show()

Output_train = hybnet(Specs_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTrainLoss = HybridNet.MatchLossFcn(Specs_train[0, :].to(device_test), Output_train)
plt.figure()
plt.plot(WL, Specs_train[0, :].cpu().numpy())
plt.plot(WL, Output_train.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path / 'train')
plt.show()

Output_test = hybnet(Specs_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTestLoss = HybridNet.MatchLossFcn(Specs_test[0, :].to(device_test), Output_test)
plt.figure()
plt.plot(WL, Specs_test[0, :].cpu().numpy())
plt.plot(WL, Output_test.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path / 'test')
plt.show()

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure()
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path / 'loss')
plt.show()

params_fig, params_axis = hybnet.plot_params()
params_fig.savefig(path / 'params.png')
params_fig.show()

(path / 'params_history').mkdir(parents=True, exist_ok=True)
params_history = np.array(params_history)
params_history = params_history.transpose(1,0,2)
n_TF = params_history.shape[0]

for fidx, params in enumerate(params_history[:,:,:]):
    plt.figure(figsize=(10, 5))

    c,l=[],[]
    for layer in range(params.shape[1]):
        color = 'g' if layer%2 else 'orange'
        c.append(color)
        l.append(params[:,layer])

    plt.stackplot(np.arange(params.shape[0]),*l,colors=c)
    plt.title(f'filter {fidx+1}')
    plt.savefig(path / 'params_history' / f'filter_{fidx+1}.png')
    plt.close()
