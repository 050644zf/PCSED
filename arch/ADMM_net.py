import torch
import torch.nn as nn
import torch.nn.functional as F
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class ADMM_net(nn.Module):

    def __init__(self, shape=(128,128,121), stage=9):
        super(ADMM_net, self).__init__()

        self.shape = shape
        self.h, self.w, self.c = shape
        self.stage = stage
        self.unets = []
        self.gammas = []
        self.unets = torch.nn.ModuleList([Unet(self.c, self.c) for _ in range(self.stage)])
        self.gammas = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor([0])) for _ in range(self.stage)])


    def forward(self, y, input_mask=None):
        if input_mask == None:
            Phi = torch.rand((1, self.shape[2], self.shape[0], self.shape[1])).cuda() # 3d mask
            Phi_s = torch.rand((1, self.shape[0], self.shape[1])).cuda()    # 2d mask
        else:
            Phi, Phi_s = input_mask
        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)

        for i in range(self.stage):
            yb = A(theta+b,Phi)
            x = theta+b + At(torch.div(y-yb,Phi_s+self.gammas[i]),Phi)
            x1 = x-b
            # x1 = shift_back_3d(x1)
            theta = self.unets[i](x1)
            # theta = shift_3d(theta)
            b = b- (x-theta)
            x_list.append(theta)
        return theta[:, :, :, 0:self.shape[0]]

# class ADMM_net_2D(nn.Module):
#     def __init__(self, n_channel=121,stages=9):
#         self.gammas = torch.nn.Parameter(torch.Tensor([0]*stages))
#         super(ADMM_net_2D, self).__init__()
#         self.stages = stages
#         self.n_channel = n_channel

#     def forward(self, y, Phi):
        

class ADMM_net_2D(nn.Module):
    def __init__(self, n_channel=121,stages=9):
        super(ADMM_net_2D, self).__init__()
        self.gammas = torch.nn.Parameter(torch.Tensor([10]*stages))
        self.stages = stages
        self.n_channel = n_channel

    def forward(self, y, Phi:torch.Tensor):
        # y: [batch, 9]
        # Phi: [9, 121]
        def At(y,p):
            temp = y.unsqueeze(1).repeat(1,p.shape[1],1)
            x = temp*p
            return x
        
        PhiT = Phi.transpose(1,0).unsqueeze(0) # [1, 121, 9]

        x_list = []
        theta = At(y,PhiT).sum(dim=2) # PhiT @ y [batch, 121]
        b = torch.zeros_like(theta)  

        Phi_PhiT = torch.matmul(Phi,PhiT[0])
        rm = torch.diag(Phi_PhiT) 

        for i in range(self.stages):
            yb = torch.matmul(theta+b,PhiT[0])
            x = theta+b      
            inv = torch.div(y-yb,rm+self.gammas[i]) # [batch, 9]
            x += At(inv,PhiT).sum(dim=2)
            theta = x-b
            b = b- (x-theta)
            x_list.append(theta)

        return theta
    
class MyModel(nn.Module):
    def __init__(self,channel=26):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(channel, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, channel)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(x.size(0), x.size(1), 1)
        return x

class ADMM_net_2(nn.Module):
    def __init__(self, channel=26, SNR=30, alpha=0, stages=9):
        super(ADMM_net_2, self).__init__()
        self.mymodel = MyModel(channel)
        self.stages = stages
        # merge all gamma into one
        self.gamma = torch.nn.Parameter(torch.Tensor([10]*stages))
    def forward(self, imgs, Phi): #, Train=True):
        # if Train:
        #     self.noisemodel.quantization=False
        #     self.noisemodel.add_noise=False
        # else:
        #     self.noisemodel.quantization=True
        #     self.noisemodel.add_noise=True

        size = imgs.size()
        batch, Nx, _ = size

        Phi = Phi.float()
        PhiT = torch.transpose(Phi.float(), dim0=0, dim1=1)
        PhiT_Phi = torch.matmul(PhiT, Phi)

        clear_y = torch.matmul(Phi, imgs)
        y = clear_y
        v = torch.matmul(PhiT,y)
        u = torch.zeros_like(v)
        I = torch.eye(PhiT_Phi.size()[0]).to(PhiT_Phi.device)
        for i in range(1, self.stages+1):
            # ************   方法一求x  直接法求逆   ******************
            gamma = self.gamma[i-1]
            qiu_ni = torch.inverse(PhiT_Phi+(gamma*I))
            vplusu = v+u
            x = torch.matmul(qiu_ni, (torch.matmul(PhiT, y) + vplusu))
            v = self.mymodel(x - u)
            u = u - (x - v)
            # ************   方法二求 x 通过公式替代求逆   ***************
            # rm = torch.diag(torch.matmul(Phi, PhiT))
            # up = y - torch.matmul(Phi, ((v+u)/self.gamma[i-1]))
            # down = (self.gamma[i-1]+rm).reshape(9,1)
            # at = up / down
            # xt = (v+u)/self.gamma[i-1] + torch.matmul(PhiT, at)
            # v = self.mymodel(xt - u)
            # u = u - (xt - v)
            # ****************以下为算保真率代码****************************
            # x_p = x.squeeze()
            # xt_p = xt.squeeze()
            # x_p = x_p.detach().cpu().numpy()
            # xt_p = xt_p.detach().cpu().numpy()
            # fidelity_pixel = np.dot(x_p, xt_p) / (np.linalg.norm(x_p) * np.linalg.norm(xt_p))
            # print("site: ", i, ". fidelity: ", fidelity_pixel)
        return v