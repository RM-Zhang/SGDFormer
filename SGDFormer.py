    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
from natten import NeighborhoodAttention2D
sys.path.append("./")
from libs.models_deformable_detr.ops.modules import * 


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAF_UNet(nn.Module):
    def __init__(self, img_channel = 1, width = 32):
        super(NAF_UNet, self).__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encode_1 = NAFBlock(c = width)   # H * W * c
        self.down_1 = nn.Conv2d(1*width, 2*width, 2, 2)
        self.encode_2 = NAFBlock(c = width*2) # H/2 * W/2 * 2c
        self.down_2 = nn.Conv2d(2*width, 4*width, 2, 2)
        self.encode_3 = NAFBlock(c = width*4) # H/4 * W/4 * 4c
        self.decode_3 = NAFBlock(c = width*4) # H/4 * W/4 * 4c
        self.up_2 = nn.Sequential(nn.Conv2d(width*4, width*8, 1, bias=False), nn.PixelShuffle(2))
        self.decode_2 = NAFBlock(c = width*2) # H/2 * W/2 * 2c
        self.up_1 = nn.Sequential(nn.Conv2d(width*2, width*4, 1, bias=False), nn.PixelShuffle(2))
        self.decode_1 = NAFBlock(c = width*1) # H * W * c
    
    def forward(self, input):
        input = self.intro(input)
        # encoder
        f_en_1 = self.encode_1(input)
        f_en_2 = self.down_1(f_en_1)
        f_en_2 = self.encode_2(f_en_2)
        f_en_3 = self.down_2(f_en_2)
        f_en_3 = self.encode_3(f_en_3)
        # decoder
        f_de = self.decode_3(f_en_3)
        f_de = self.up_2(f_de)
        f_de = self.decode_2(f_de + f_en_2)
        f_de = self.up_1(f_de)
        f_de = self.decode_1(f_de + f_en_1)
        return f_de

  
class SVFF(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()

        self.conv_A = nn.Sequential(nn.Conv2d(in_channels=d_model*2, out_channels=d_model, kernel_size=3, padding=1, stride=1), nn.Tanh())
        self.conv_B = nn.Sequential(nn.Conv2d(in_channels=d_model*2, out_channels=d_model, kernel_size=3, padding=1, stride=1), nn.Tanh())
        self.conv_F = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1), nn.GELU(), LayerNorm2d(d_model))
        
    def forward(self, noisy, guidance):
        feature = torch.cat([noisy, guidance], dim = 1)
        A_map = self.conv_A(feature)
        B_map = self.conv_B(feature)
        fusion = noisy * B_map + guidance * A_map
        fusion = self.conv_F(fusion)
        return fusion


class Refine_Attn(nn.Module):
    def __init__(self, max_disp=128, local_range=5):
        super().__init__()
        
        self.local_range = local_range
        self.max_disp = max_disp
        
        self.na2d_1 = NeighborhoodAttention2D(dim=max_disp//4, kernel_size=self.local_range, dilation=1, num_heads=1)
        self.up_1 = nn.Sequential(torch.nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(max_disp//4, max_disp//2, 1, 1))
        self.na2d_2 = NeighborhoodAttention2D(dim=max_disp//2, kernel_size=self.local_range, dilation=1, num_heads=1)
        self.up_2 = nn.Sequential(torch.nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(max_disp//2, max_disp, 1, 1))
        self.na2d_3 = NeighborhoodAttention2D(dim=max_disp, kernel_size=self.local_range, dilation=1, num_heads=1)

    def forward(self, attn):
        attn = self.na2d_1(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = self.up_1(attn)
        attn = self.na2d_2(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = self.up_2(attn)
        attn = self.na2d_3(attn.permute(0,2,3,1)).permute(0,3,1,2)
        return attn

    
class NRCA(nn.Module):
    def __init__(self, d_model=32, max_disp=128, local_range=5):
        super().__init__()
        
        self.max_disp = max_disp
        self.q = nn.Sequential(nn.Conv2d(1*d_model, 2*d_model, 2, 2), nn.GELU(), LayerNorm2d(2*d_model),
                                    nn.Conv2d(2*d_model, 2*d_model, 2, 2), nn.GELU(), LayerNorm2d(2*d_model))
        self.k = nn.Sequential(nn.Conv2d(1*d_model, 2*d_model, 2, 2), nn.GELU(), LayerNorm2d(2*d_model),
                                    nn.Conv2d(2*d_model, 2*d_model, 2, 2), nn.GELU(), LayerNorm2d(2*d_model))
        self.refine_attn = Refine_Attn(max_disp=max_disp, local_range=local_range)
        self.aggregate_f = MSDeformAttn_SGDFormer(d_model, 1, 1)
        
    def forward(self, noisy, guidance):
        
        f_q = self.q(noisy)
        f_k = self.k(guidance)

        attn = build_correlation_volume(f_q, f_k, self.max_disp//4) #[B,1,D,H,W]
        attn = torch.squeeze(attn, dim=1) #[B,D,H,W]
        attn = self.refine_attn(attn)
        attn = F.softmax(attn, dim=1)
        index = get_ref_max_points(attn)

        # aggregate f
        D = attn.shape[1]
        spatial_shapes = []
        bs, c, h, w = noisy.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=noisy.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        attn = attn.transpose(1, 2).transpose(2, 3).reshape(bs, h*w, 1, 1, D).contiguous()
        index = index.transpose(1, 2).transpose(2, 3).reshape(bs, h*w, 1, 1, D, 2).contiguous()
        guidance = guidance.flatten(2).transpose(1, 2)
        f_gw = self.aggregate_f(guidance, spatial_shapes, level_start_index, index, attn)
        f_gw = f_gw.permute(0,2,1).reshape(bs, c, h, w)
        
        return f_gw

    
class TransBlock(nn.Module):
    def __init__(self, d_model=32, max_disp=128, local_range=5):
        super().__init__()
        self.nrca = NRCA(d_model=d_model, max_disp=max_disp, local_range=local_range)
        self.svff = SVFF(d_model=d_model)
        self.ffn = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm = LayerNorm2d(d_model)

    def forward(self, noisy, guidance):
        f_gw = self.nrca(noisy, guidance)
        f_f = self.svff(noisy, f_gw)
        f_f = self.norm(self.ffn(f_f)) + f_f
        return f_f


def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous() # [B,1,max_disp,H,W]
    return volume


def get_ref_max_points(corr):
    # corr: [B, max_disp, H, W]
    # return: [B, max_disp, H, W, 2], value [0,1]
    [B, C, H, W] = corr.shape

    ref_y, ref_x = torch.meshgrid(torch.linspace(0, H-1, H, dtype=torch.float32, device = corr.device),
                                  torch.linspace(0, W-1, W, dtype=torch.float32, device = corr.device))
    ref = torch.stack((ref_x, ref_y), -1)
    ref = ref.unsqueeze(0).unsqueeze(0) # [1,1,H,W,2]
    ref = ref.repeat(B, C, 1, 1, 1) # [B,C,H,W,2]
    
    disp = torch.linspace(0, C-1, C, dtype=torch.float32, device = corr.device)
    disp = disp.unsqueeze(0).unsqueeze(2).unsqueeze(3) # [1,C,1,1]
    disp = disp.repeat(B, 1, H, W)
    
    ref[:,:,:,:,0] = ref[:,:,:,:,0] - disp
    ref[:,:,:,:,0] = ref[:,:,:,:,0].clone()/max(W-1,1)
    ref[:,:,:,:,1] = ref[:,:,:,:,1].clone()/max(H-1,1)
    
    return ref


class SGDFormer(nn.Module):
    def __init__(self, img_channel = 1, width = 32, max_disp = 128, local_range = 5, trans_num = 1):
        super(SGDFormer, self).__init__()
        
        self.max_disp = max_disp
        self.local_range = local_range
        self.trans_num = trans_num
        
        self.encoder_n = NAF_UNet(width=width)
        self.encoder_g = NAF_UNet(width=width)
        
        self.trans_list = nn.ModuleList([TransBlock(d_model=width, max_disp=self.max_disp, local_range=self.local_range) for _ in range(self.trans_num)])
        
        self.ending_n = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
    def forward(self, noisy, guidance):
        
        f_de_n = self.encoder_n(noisy)
        f_de_g = self.encoder_g(guidance)

        for i, trans in enumerate(self.trans_list):
            f_de_n = trans(f_de_n, f_de_g)
        
        res = self.ending_n(f_de_n)
        return res
    

from thop import profile 
if __name__ == "__main__":
    
    net = SGDFormer(img_channel=1, width=32, max_disp=128, local_range=5, trans_num=1).cuda()
    noisy = torch.randn(1,1,256,256).cuda()
    guidance = torch.randn(1,1,256,256).cuda()
    macs, params = profile(net, inputs=(noisy, guidance))
    print('FLOPs = ' + str(macs /1000**3) + 'G')
    print('Params = ' + str(params /1000**2) + 'M')
