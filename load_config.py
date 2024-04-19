# Import necessary libraries
import h5py
import torch
import scipy.io as scio
import numpy as np
import time
import os
import json
import shutil
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml', help='path to config file')
parser.add_argument('-n', '--nettype', type=str, default='hybnet', help='type of network to train')
parser.add_argument('-r','--response',type=str, default='',help='folder to alter the response curves')
parser.add_argument('--pretained',type=str, default='./results/wayho/model/model_epoch_279.pth',help='path to the pretained model')
parser.add_argument
args = parser.parse_args()

# Set working directory to the directory of this script
os.chdir(Path(__file__).parent)

# Load configuration from YAML file
import yaml

if args.nettype == 'ADMM_Net':
    with open('config_ADMM.yml', 'r') as f:
        c = yaml.safe_load(f)
        config: dict = c['PCSED']
        noise_config = c['noise']
        if args.nettype == 'ADMM_Net':
            admm_config = c['ADMM_Net']
else:
    with open('config.yml', 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
        config: dict = c['PCSED']
        noise_config = c['noise']
        if args.nettype == 'ADMM_Net':
            admm_config = c['ADMM_Net']

# Set data type and device for data and training
dtype = torch.float
device_data = torch.device("cpu")
device_train = torch.device("cuda:0")
device_test = torch.device("cuda:0")

# Set parameters from configuration
Material = 'TF'
TrainingDataSize = config['TrainingDataSize']
TestingDataSize = config['TestingDataSize']
BatchSize = config['BatchSize']
EpochNum = config['EpochNum']
TestInterval = config['TestInterval']
lr = config['lr']
lr_decay_step = config['lr_decay_step']
lr_decay_gamma = config['lr_decay_gamma']
beta_range = config['beta_range']
TFNum = config['TFNum']
thickness_error = config['thickness_error']
total_thickness_range = config['total_thickness_range']
# Create folder to save trained HybNet
folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())

path = Path(f'nets/{args.nettype}/{folder_name}/')
path.mkdir(parents=True, exist_ok=True)

# Load configuration for fnet
fnet_folder = Path(config['fnet_folder'])
with open(fnet_folder/'config.json',encoding='utf-8') as f:
    fnet_config = json.load(f)['fnet']

# Save configuration for HybNet and fnet
with open(path/'config.yml', 'w', encoding='utf-8') as f:
    yaml.dump(
        {'fnet': fnet_config,'PCSED': config}
        , f, default_flow_style=False)
shutil.copy(fnet_folder/'n.mat',path/'n.mat')
shutil.copy('arch/HybridNet.py',path/'HybridNet.py')


# Load fnet
fnet_path = fnet_folder/'fnet.pkl'
params_min = torch.tensor([fnet_config['params_min']])
params_max = torch.tensor([fnet_config['params_max']])

# Set wavelength range and number of spectral slices
StartWL = fnet_config['StartWL']
EndWL = fnet_config['EndWL']
Resolution = fnet_config['Resolution']
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size

if config.get('LightPath'):
    LightMat = scio.loadmat(config['LightPath'])['data']
    LightNum = LightMat.shape[0]
else:
    LightMat = np.ones([1, SpectralSliceNum], dtype=np.float32)
    LightNum = LightMat.shape[0]

if args.nettype == 'ADMM_Net':
    pass
else:
    # Load training and testing data


    LightMat = torch.tensor(LightMat, device=device_data, dtype=dtype)
    with h5py.File(config['TrainDataPath'], 'r') as file:
        Specs_all = file['combined_array'][:].T
    # data = scio.loadmat(config['TrainDataPath'])
    # Specs_all = np.array(data['data'])

    non_zero_indices = np.nonzero(np.sum(Specs_all, axis=1))
    Specs_all = Specs_all[non_zero_indices]

    TrainingDataSize = min(TrainingDataSize, Specs_all.shape[0])
    np.random.shuffle(Specs_all)
    Specs_all = torch.tensor(Specs_all[0:TrainingDataSize, :])
    Specs_train = torch.zeros([TrainingDataSize * LightNum, SpectralSliceNum], device=device_data, dtype=dtype)
    for i in range(LightNum):
        Specs_train[i * TrainingDataSize: (i + 1) * TrainingDataSize, :] = Specs_all *   LightMat[i, :]
    
    # LightMat = torch.tensor(LightMat, device=device_test, dtype=dtype)
    LightMat = LightMat.clone().detach().to(device_test).type(dtype)
    data = scio.loadmat(config['TestDataPath'])
    Specs_all = np.array(data['data'])

    non_zero_indices = np.nonzero(np.sum(Specs_all, axis=1))
    Specs_all = Specs_all[non_zero_indices]

    TestingDataSize = min(TestingDataSize, Specs_all.shape[0])
    np.random.shuffle(Specs_all)
    Specs_all = torch.tensor(Specs_all[0:TestingDataSize, :], device=device_test)
    Specs_test = torch.zeros([TestingDataSize * LightNum, SpectralSliceNum], device=device_test, dtype=dtype)
    for i in range(LightNum):
        Specs_test[i * TestingDataSize: (i + 1) * TestingDataSize, :] = Specs_all * LightMat[i, :]
    del Specs_all, data

print(f'SpectralSliceNum: {SpectralSliceNum} TrainingDataSize: {TrainingDataSize} TestingDataSize: {TestingDataSize} LightNum: {LightNum}')

TrainingDataSize = TrainingDataSize * LightNum
TestingDataSize = TestingDataSize * LightNum

# Check that the number of spectral slices matches the size of the training data
assert SpectralSliceNum == Specs_train.size(1)

# Load QEC data if specified in configuration
QEC = 1
if config.get('QEC'):
    QEC = scio.loadmat(config['QEC'])['data']