import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.registry import register_model
from registry import register_pip_model
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import os
import math
import copy
from functools import partial
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from utils import selective_scan_flop_jit, flops_selective_scan_fn
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        padding = 1
        stride = patch_size // 2
        kernel= stride + 1 
        hidden_dim = embed_dim//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim , kernel_size=kernel, stride=stride, padding=padding),
            LayerNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=kernel, stride=stride, padding=padding),
            LayerNorm2d(embed_dim),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(dim_out)   
        )
     
    def forward(self, x):
        x = self.down(x)
        return x


class ConvFFN(nn.Module):
    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()
        self.in_features = channels
        self.hidden_features = channels * expansion
        self.out_features =  channels

        self.fc1 = nn.Conv2d(self.in_features, self.hidden_features, 1, 1, 0)
        self.act = nn.GELU()
        self.dwConv = nn.Conv2d(self.hidden_features, self.hidden_features, 3, 1, 1, groups=self.hidden_features)
        self.fc2 = nn.Conv2d(self.hidden_features, self.out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dwConv(x)
        x = self.act(x)
        x = self.drop(x)
        x= self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SSM(nn.Module):
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
            
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, device=None):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, device=None):
        D = nn.Parameter(torch.ones(d_inner, device=device)) 
        D._no_weight_decay = True
        return D
    
    def __init__(
        self,
        d_model=94,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__() 
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank  
        self.out_norm = LayerNorm2d(self.d_inner)
        self.d_proj = self.d_inner * 2
        self.in_proj = nn.Conv2d(d_model, self.d_proj, kernel_size=1, bias=False)
        self.act = nn.SiLU()

        d_conv = 3
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + d_state * 2), bias=False)
        self.out_proj = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        self.A_log = self.A_log_init(d_state, self.d_inner) 
        self.D = self.D_init(self.d_inner) 
        
        self.total_scan = 4
        self.weights = nn.Parameter(torch.randn(4))

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.d_inner, 1, 1, 1).to(torch.device("cuda"))
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.d_inner, 1, 1, 1).to(torch.device("cuda"))
        self.filter_custom = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, stride=1, groups=self.d_inner),
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size = 1),
            nn.ReLU()
        )
        self.filter_vert = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, stride=1, groups=self.d_inner),
            nn.ReLU()
        )

        self.filter_hori = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, stride=1, groups=self.d_inner),
            nn.ReLU()
        )
        
        
    def merge(self, x):
        x_o = x.flatten(2)
        vert_filter = F.conv2d(x, self.sobel_x, padding='same', groups=x.shape[1])
        vert_filter =  self.filter_vert(vert_filter)
        x_vert_filter = vert_filter.flatten(2).contiguous()

        x_auto_filter = self.filter_custom(x).flatten(2).contiguous()

        hori_filter = F.conv2d(x, self.sobel_y, padding='same', groups=x.shape[1])
        hori_filter = self.filter_hori(hori_filter)
        x_hori_filter = hori_filter.flatten(2).contiguous()
        return torch.cat( [x_o, x_vert_filter, x_auto_filter, x_hori_filter], dim=2).contiguous()


    def split(self, x, H, W):
        b, _, _ = x.shape
        x_parts = x.chunk(self.total_scan, dim=2)
        x_o =  x_parts[0].view(b, -1, H, W)
        x_vert_filter = x_parts[1].view(b, -1, H, W)
        x_auto_filter = x_parts[2].view(b, -1, H, W)
        x_hori_filter = x_parts[3].view(b, -1, H, W)
        wt = F.softmax(self.weights, dim=0)
        return (wt[0] * x_o + wt[1] * x_vert_filter + wt[2] * x_auto_filter + wt[3] * x_hori_filter).contiguous()

    
    def state_operation(self, x, force_fp32=True, delta_softplus = True):
        _, N = self.A_log.shape  
        R = self.dt_rank
        _batch, _, H, W = x.shape
        L = H * W * self.total_scan
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
         
        x = self.merge(x)  
        x_dbl = self.x_proj(rearrange(x, " b d l -> (b l) d"))  
        dt, B, C = torch.split(x_dbl, [R, N, N], dim = -1)
        dt = self.dt_proj(dt) 
        dt = rearrange(dt, "(b l) d -> b d l", l=L)
        x = x.view(_batch, -1, L).contiguous() 
        B = rearrange(B, "(b l) n -> b n l", l=L).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=L).contiguous()
        A = -torch.exp(self.A_log.to(torch.float))
        D = self.D.to(torch.float)
        delta_bias = self.dt_proj.bias.to(torch.float)

        if force_fp32:
            x, dt, B, C = to_fp32(x, dt, B, C)

        y: torch.Tensor = selective_scan_fn( x, dt, A, B, C, D, None, delta_bias, delta_softplus)
        y = self.split(y, H, W)  
        y = y.view(_batch, -1, H, W)
        y = self.out_norm(y)
        return y.to(x.dtype) 

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=1) 
        z = self.act(z)
        x = self.conv2d(x)
        x = self.act(x)
        y = self.state_operation(x = x, force_fp32=True)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        drop_path,
        ssm_d_state,
        ssm_ratio,
        ssm_dt_rank,
        mlp_ratio
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(hidden_dim)

        self.ssm = SSM(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            dropout=0,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
        )
       
        self.drop_path = DropPath(drop_path)
        self.convFFN = ConvFFN(hidden_dim, mlp_ratio)
        self.norm2 = LayerNorm2d(hidden_dim)

    def forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.ssm(self.norm1(x)))
        x = x + self.drop_path(self.norm2(self.convFFN(x)))
        return x

class MambaLayer(nn.Module):
    def __init__(self, 
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        downsample=nn.Identity(),
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([ 
            MambaBlock(
                hidden_dim=dim, 
                drop_path=drop_path[i],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                mlp_ratio=mlp_ratio,
            ) for i in range(depth)] )
        
        self.downsample = downsample

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return x


class MFilMamba(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 3, 8, 2], 
        dims= 94, 
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        mlp_ratio=4.0,
        drop_path_rate=0.1, 
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
     
        self.patch_embed = PatchEmbed(in_chans= in_chans, embed_dim=dims[0], patch_size= patch_size )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer(  
                dim = self.dims[i_layer],
                depth= depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=Downsample(dim = self.dims[i_layer], dim_out=self.dims[i_layer + 1]) if  (i_layer < self.num_layers - 1) else nn.Identity(),
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                mlp_ratio=mlp_ratio,
                )
            self.layers.append(layer)

        self.norm = LayerNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        supported_ops={
            "aten::silu": None, 
            "aten::neg": None, 
            "aten::exp": None, 
            "aten::flip": None, 
            "prim::PythonOp.SelectiveScanFn": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn)
            }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e6')
            return 1e6
        del model, input
        return sum(Gflops.values()) * 1e9


@register_pip_model
@register_model
def mfil_tiny(pretrained=False, **kwargs):
    model = MFilMamba(
                patch_size=4, 
                in_chans=3, 
                num_classes=1000, 
                depths=[1, 3, 8, 2], 
                dims=94, 
                ssm_d_state=1,
                ssm_ratio=1.0,
                ssm_dt_rank="auto",
                mlp_ratio=4,
                drop_path_rate=0.2,
            )
    return model


@register_pip_model
@register_model
def mfil_small(pretrained=False, **kwargs):
    model = MFilMamba(
                patch_size=4, 
                in_chans=3, 
                num_classes=1000, 
                depths=[2, 2, 18, 2], 
                dims = 94,
                ssm_d_state=1,
                ssm_ratio=1.0,
                ssm_dt_rank="auto",
                mlp_ratio=4,
                drop_path_rate=0.3,
            )
    return model


@register_pip_model
@register_model
def mfil_base(pretrained=False, **kwargs):
    model = MFilMamba(
                patch_size=4, 
                in_chans=3, 
                num_classes=1000, 
                depths=[2, 2, 18, 2],
                dims=128, 
                ssm_d_state=1,
                ssm_ratio=1.0,
                ssm_dt_rank="auto",
                mlp_ratio=4,
                drop_path_rate=0.5,
            )
    return model

