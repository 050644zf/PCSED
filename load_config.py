# Import necessary libraries
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
parser.add_argument('--pretained',type=str, default='',help='path to the pretained model')
parser.add_argument
args = parser.parse_args()

# Set working directory to the directory of this script
os.chdir(Path(__file__).parent)

# Load configuration from YAML file
import yaml
with open('config.yml', 'r') as f:
    config: dict = yaml.safe_load(f)['PCSED']

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

if args.nettype != 'ADMM_Net':
    # Load training and testing data
    Specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
    Specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
    data = scio.loadmat(config['TrainDataPath'])
    Specs_all = np.array(data['data'])
    np.random.shuffle(Specs_all)
    Specs_train = torch.tensor(Specs_all[0:TrainingDataSize, :])
    data = scio.loadmat(config['TestDataPath'])
    Specs_all = np.array(data['data'])
    np.random.shuffle(Specs_all)
    Specs_test = torch.tensor(
        Specs_all[0:TestingDataSize, :], device=device_test)
    del Specs_all, data

# Check that the number of spectral slices matches the size of the training data
    assert SpectralSliceNum == Specs_train.size(1)

# Load QEC data if specified in configuration
QEC = 1
if config.get('QEC'):
    QEC = scio.loadmat(config['QEC'])['data']