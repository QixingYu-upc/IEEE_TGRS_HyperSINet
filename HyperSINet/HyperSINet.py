import math
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import data_reader
from sklearn.decomposition import PCA
from einops import rearrange#改变张量
from torch import nn
import torch.nn.init as init
from torchsummaryX import summary
# from visualize import visualize_grid_attention_v2
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

path_model = r"/home/project/HyperSINet/model1\\"

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv1 = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_qkv2 = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv1(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3 三个维度都是([64,5,64])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions，分割为多头注意力，q,k,v的维度分别为([64,8,5,8])
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * mask
        # v = torch.einsum('ij,bhjk->bhik',mask,v)
        dots = nn.Sigmoid()(dots)
        out1 = torch.einsum('bhij,bhjd->bhid',dots,v)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        qkv = self.to_qkv2(out1).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3 三个维度都是([64,5,64])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),qkv)  # split into multi head attentions，分割为多头注意力，q,k,v的维度分别为([64,8,5,8])
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # ([64,8,5,5])
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper ([64,8,5,5])
        # np.save('att.npy',attn.cpu().detach().numpy())
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax ([64, 8, 5, 8])
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block ([64, 5, 64])
        out = self.nn1(out)# ([64, 5, 64])
        out = self.do1(out)# ([64, 5, 64])
        return out

class SSConv(nn.Module):
    '''Spectral-Spatial Convolution'''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class Channel_only_branch(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out
class Spatial_only_branch(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        return spatial_out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out
class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CNN_(nn.Module):
    def __init__(self,in_channels):
        super(CNN_, self).__init__()
        self.c1 = Spatial_only_branch(in_channels)
        self.c2 = Spatial_only_branch(in_channels)
        self.s1 = Channel_only_branch(in_channels)
        self.s2 = Channel_only_branch(in_channels)
        self.ss1 = SSConv(in_channels,in_channels,kernel_size=3)
        self.ss2 = SSConv(in_channels,in_channels,kernel_size=5)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self._2x1 = nn.Conv2d(in_channels*2,in_channels*2,(2,1))
        self._1x1 = nn.Conv2d(in_channels*2,in_channels,1)
        self.Ac = nn.Sigmoid()
    def forward(self,x):
        xx = x
        x = self.ss1(x)
        spa = self.c1(x)
        spe = self.s1(x)
        out = torch.cat([spa,spe],dim=1)
        avg = self.avg(out)
        _max = nn.MaxPool2d((x.shape[2],x.shape[3]))(out)
        cat = torch.cat([avg,_max],dim = 2)
        cat = self._2x1(cat)
        cat = self.Ac(cat)
        out = cat * out + out
        out = self._1x1(out)
        x1 = self.c2(out)
        x2 = self.s2(out)
        x = self.ss2(x1 + x2 + xx)
        return x
class Trans2Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Trans2Conv, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels,eps=1e-6)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.ac = nn.LeakyReLU(inplace=True)
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.nn1 = nn.Linear(in_channels, in_channels // 2)
        self.nn2 = nn.Linear(in_channels // 2, in_channels)
    def forward(self,x, Q, height, width,conv_x):
        b,h,w = x.shape
        x = torch.matmul(Q,x)
        x = torch.reshape(x,(b,w,height,width))
        # x_up = self.ac(self.bn(x))
        x_up = self.bn(x)
        xg = self.global_att(x_up+conv_x)
        xg = torch.einsum('b c i i->b i c',xg)
        # print(xg.shape)
        xg = self.nn1(xg)
        xg = self.nn2(xg)
        # print(xg.shape)
        xg = torch.einsum('b i c->b c i', xg)
        xg = torch.unsqueeze(xg,2)
        xg = self.sigmoid(xg)
        xo = (1-xg) * conv_x + xg * x_up
        xo = self.ac(xo)
        # xa = x_up+conv_x
        # xl = self.local_att(xa)
        # xg = self.global_att(xa)
        # xlg = xl+xg
        # wei = self.sigmoid(xg)
        # xo = wei*conv_x+x_up*(1-wei)
        return xo
class Conv2Trans(nn.Module):
    def __init__(self,in_channels):
        super(Conv2Trans, self).__init__()
        self.sample_pooling = nn.AvgPool2d(kernel_size=1, stride=1)
        self.ln = nn.LayerNorm(normalized_shape=64)
        self.ac1 = nn.GELU()
        self.nn1 = nn.Linear(in_channels, in_channels // 2)
        # torch.nn.init.xavier_normal_(self.nn1)
        self.nn2 = nn.Linear(in_channels // 2, in_channels)
        # torch.nn.init.xavier_normal_(self.nn2)
        self.nn3 = nn.Linear(in_channels, in_channels)
        # torch.nn.init.xavier_normal_(self.nn3)
        self.ac3 = nn.GELU()
        self.ac2 = nn.Sigmoid()
    def forward(self,x,Q_cor,x_t):
        x = self.sample_pooling(x)
        x = rearrange(x,'b c h w -> b c (h w)')
        x = rearrange(x,'b h w -> b w h')
        x = torch.matmul(Q_cor.T,x)
        x = self.ln(x)
        x = self.ac1(x)
        xa = x + x_t
        x1 = self.nn1(xa)
        x2 = self.nn2(x1)
        xg = self.ac2(x2)
        x = (1-xg) * x_t + xg * x

        x = self.ac3(x)
        return x
        # xa = x+x_t
        # x1 = self.nn1(xa)
        # x2 = self.nn2(xa)
        # x2 = self.ac2(x2)
        # x1 = x1*x2
        # x = self.nn3(x1)
        # return x+xa
# class Trans2Conv(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Trans2Conv, self).__init__()
#         self.bn = nn.BatchNorm2d(out_channels,eps=1e-6)
#         self.ac = nn.ReLU()
#     def forward(self,x, Q, height, width):
#         b,h,w = x.shape
#         x = torch.matmul(Q,x)
#         x = torch.reshape(x,(b,w,height,width))
#         x_up = self.ac(self.bn(x))
#         return x_up
# class Conv2Trans(nn.Module):
#     def __init__(self,in_channels):
#         super(Conv2Trans, self).__init__()
#         self.sample_pooling = nn.AvgPool2d(kernel_size=1, stride=1)
#         self.ln = nn.LayerNorm(normalized_shape=64)
#         self.ac = nn.GELU()
#     def forward(self,x,Q_cor):
#         x = self.sample_pooling(x)
#         x = rearrange(x,'b c h w -> b c (h w)')
#         x = rearrange(x,'b h w -> b w h')
#         x = torch.matmul(Q_cor.T,x)
#         x = self.ln(x)
#         x = self.ac(x)
#         return x
class Conv_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))),
                CNN_(in_channels=dim),
                Trans2Conv(in_channels=dim, out_channels=dim),
                Conv2Trans(in_channels=dim)
            ]))
    def forward(self, conv_x, x_t, Q, Q_cor, height, width, mask = None,):
        for attention, mlp, pydw, trans2conv, conv2trans in self.layers:
            x_t = attention(x_t, mask=mask)  # go to attention
            x_t = mlp(x_t)
            conv_x = pydw(conv_x)
            t2c = trans2conv(x_t, Q, height, width,conv_x)
            c2t = conv2trans(conv_x, Q_cor,x_t)
            x_t = c2t
            # t2c = trans2conv(x_t,Q,height,width,conv_x)
            # c2t = conv2trans(conv_x,Q_cor,x_t)
            # x_t = c2t
            # x_t = x_t + c2t
            # conv_x = t2c
            conv_x = t2c
        return x_t, conv_x

NUM_CLASS = 16

class Denoise(nn.Module):
    def __init__(self,channel:int,out_channels:int,layers:int):
        super(Denoise,self).__init__()
        self.channel = channel
        self.out_channels = out_channels
        self.layers = layers
        self.CNN_denoise = nn.Sequential()
        for i in range(self.layers):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel,self.out_channels,kernel_size=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.out_channels))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
    def forward(self,x):
        return self.CNN_denoise(x)
class HyperSINet(nn.Module):
    def __init__(self, Q, mask, in_channels=1, out_channels=30, layers=2, num_classes=NUM_CLASS, num_tokens=8, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(HyperSINet, self).__init__()
        self.Denoise = Denoise(in_channels, out_channels, layers)
        self.Q = Q
        self.n_mask = mask
        self.Q_cor = Q / (torch.sum(Q, 0, keepdim=True))
        self.dropout = nn.Dropout(emb_dropout)
        self.conv_transformer = Conv_Transformer(dim, depth, heads, mlp_dim, dropout)
        self.nn1 = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, 0)
        x = self.Denoise(x)
        x_t = rearrange(x, 'b c h w -> b  (h w) c')  # x的尺度  ([64, 81, 64])
        x_t = torch.matmul(self.Q_cor.T,x_t)
        x_t, x = self.conv_transformer(x, x_t, self.Q, self.Q_cor, x.shape[2],x.shape[3],self.n_mask)
        x_t = torch.matmul(self.Q,x_t)
        x = rearrange(x,'b c h w -> b (h w) c')
        # result = x + x_t
        result = 0.05 * x_t + x * 0.95
        x = self.nn1(result)
        x = x.squeeze(0)
        return torch.softmax(x,-1)


import torchvision
from ptflops import get_model_complexity_info
if __name__ == '__main__':
    model = torchvision.models.alexnet(pretrained=False)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)