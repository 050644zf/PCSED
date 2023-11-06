import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time
import math
import os
import json
import yaml
from pathlib import Path
from tmm_acc import coh_tmm_normal_spol_spec_d
import argparse
from HybridNet import HybridNet,NoisyHybridNet,ND_HybridNet
from NoiseLayer import *

folder = Path(__file__).parent

dtype = torch.float
device_data = torch.device('cpu')
device_test = torch.device('cuda:0')
# evaluate the model by calculating the MSE of each curve
def eval_hybnet(model, Output, T, noise_layer=None):
    model.eval()
    with torch.no_grad():
        sampled = func.linear(Output, model.fnet(model.DesignParams), None)
        if noise_layer is not None:
            sampled = noise_layer(sampled)
        if isinstance(model,ND_HybridNet):
            sampled = sampled[:,model.diff_idx] - sampled[:,model.original_idx]
        Output_pred = model.SWNet(sampled)
        loss_mat = (Output_pred - Output) ** 2
        loss = torch.mean(loss_mat, dim=1)
        return loss.cpu().numpy(), Output_pred.cpu().numpy()
    
class Hybnet_folder:
    def __init__(self,model_folder:Path):
        self.model_folder = model_folder
        try:
            with open(model_folder/'config.json',encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            with open(model_folder/'config.yml',encoding='utf-8') as f:
                config = yaml.safe_load(f)

        self.fnet_cfg = config['fnet']
        self.PCSED_cfg = config['PCSED']
        self.noise_cfg = config.get('noise',None)
        self.TFNum = self.PCSED_cfg['TFNum']

        self.WL = np.arange(self.fnet_cfg['StartWL'], self.fnet_cfg['EndWL'], self.fnet_cfg['Resolution'])

        # design test begin
        self.params = scio.loadmat(model_folder/'TrainedParams.mat')['Params']
        self.TargetCurves = scio.loadmat(model_folder/'TargetCurves.mat')['TargetCurves']
        self.TargetCurves_FMN = scio.loadmat(model_folder/'TargetCurves_FMN.mat')['TargetCurves_FMN']
        self.n_array = scio.loadmat(model_folder/'n.mat')['n']

        d_array = np.zeros((self.params.shape[0],self.params.shape[1]+2))
        d_array += np.inf
        d_array[:,1:-1] = self.params

        self.T = coh_tmm_normal_spol_spec_d(self.n_array, d_array, self.WL)


    def plot_T(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.TFNum):
            plt.subplot(math.ceil(math.sqrt(self.TFNum)), math.ceil(math.sqrt(self.TFNum)), i + 1)
            plt.plot(self.WL, self.TargetCurves[i, :])
            plt.plot(self.WL, self.TargetCurves_FMN[i, :])
            plt.plot(self.WL, self.T[i, :])
            plt.ylim(0, 1)

        plt.savefig(self.model_folder/'TargetCurves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def load_model(self,device):
        self.model = torch.load(self.model_folder/'hybnet.pkl', map_location=device)
        self.model.eval()

        if isinstance(self.model,NoisyHybridNet):
            self.noise_layer = self.model.noise_layer
        else:
            self.noise_layer = None

    def change_noise_layer(self,SNR,alpha,bitdepth=8):
        self.noise_layer = NoiseLayer(SNR=SNR, alpha=alpha, bitdepth=bitdepth)
        if isinstance(self.model,NoisyHybridNet):
            self.model.noise_layer = self.noise_layer
        elif isinstance(self.model,HybridNet):
            self.model.noise_layer = self.noise_layer
            self.model.forward = NoisyHybridNet.forward
            self.model.run_swnet = NoisyHybridNet.run_swnet

    def eval(self,data,T,device):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=device, dtype=dtype)
        data = data.to(device)
        T = torch.tensor(T, device=device, dtype=dtype)
        loss, pred = eval_hybnet(self.model, data, T, self.noise_layer)
        return loss, pred


    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str, help='path to the Hybnet model folder')
    parser.add_argument('-c','--curves',type=str, default='',help='folder to alter the curves')

    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    
    testing_model = Hybnet_folder(model_folder)

    if not args.curves == '':
        curves_folder = Path(args.curves)
        testing_model.TargetCurves = scio.loadmat(curves_folder/'TargetCurves.mat')['TargetCurves']
        testing_model.TargetCurves_FMN = scio.loadmat(curves_folder/'TargetCurves_FMN.mat')['TargetCurves_FMN']

    testing_model.load_model(device_test)
    testing_model.plot_T()
    data = scio.loadmat(testing_model.PCSED_cfg['TestDataPath'])['data']
    pred_loss, pred_output = testing_model.eval(data,testing_model.TargetCurves_FMN,device_test)
    simu_loss, simu_output = testing_model.eval(data,testing_model.T,device_test)

    logger = open(testing_model.model_folder/'test_log.txt','w+')

    print('Mean MSE of predicted curves:', np.mean(pred_loss))
    print('Mean MSE of predicted curves:', np.mean(pred_loss),file=logger)
    print('Mean MSE of simulated curves:', np.mean(simu_loss))
    print('Mean MSE of simulated curves:', np.mean(simu_loss),file=logger)

    test_SNR = [60,30,20,10,0]

    for SNR in test_SNR:
        nl = NoiseLayer(SNR=SNR, alpha=1, bitdepth=8)
        testing_model.change_noise_layer(SNR=SNR,alpha=1,bitdepth=8)
        pred_loss, pred_output = testing_model.eval(data,testing_model.TargetCurves_FMN,device_test)
        simu_loss, simu_output = testing_model.eval(data,testing_model.T,device_test)
        print(f'SNR: {SNR}, Pred_loss: {np.mean(pred_loss):.6f}, Simu_loss: {np.mean(simu_loss):.6f}')
        print(f'SNR: {SNR}, Pred_loss: {np.mean(pred_loss):.6f}, Simu_loss: {np.mean(simu_loss):.6f}',file=logger)







    




