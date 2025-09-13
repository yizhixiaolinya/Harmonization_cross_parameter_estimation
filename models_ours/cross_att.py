import torch
import torch.nn as nn
import torch.nn.functional as F
from models_ours.linear import Linear
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False):
        super(Mlp, self).__init__()

        self.project_in = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features // 2, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class C_Cross_Attention3D(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=12,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim , kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, C, H, W = x.shape #x_.shape=(B, 64, 1024)
        N = H * W
        _, C_, H_, W_ = y.shape
        N_ = H_ * W_
        q = self.q(y).reshape(B, N_, self.num_heads, C_//self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C_//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, 1, 1)
        x = self.proj(x)
        #x = self.proj_drop(x)
        return x

class Block3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_head_dim=None,
        LayerNorm_type='WithBias'
    ):
        super().__init__()
        #self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.c_attn = C_Cross_Attention3D(
        dim,
        num_heads=num_heads,
        qk_scale=qk_scale,
        attn_drop=attn_drop,
        proj_drop=drop,
        attn_head_dim=attn_head_dim
        )
        
        self.text_lora = Linear(in_dim=dim, out_dim=dim, hidden_list = [dim])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim
        )

    def forward(self, x, y):
        if y.size(1) != 1:
            y = y.mean(dim=1)  
        y = self.text_lora(y.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
        # print('y', y.shape,flush=True) # [B, 768, 1, 1]
        # print('x before c_attn', x.shape,flush=True) # [B, 768, 256, 256] 
        x = x * self.c_attn(x, y)
        # print('x after c_attn', x.shape,flush=True) # [B, 768, 256, 256]
        x = self.norm2(x)
        # print('x after norm2', x.shape,flush=True)
        x = x + self.drop_path(self.mlp(x))
        # print('x after mlp', x.shape,flush=True)
        
        return self.norm3(x)

class Basic_block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads
    ):
        super().__init__()
        self.depth = 1
        self.block = nn.ModuleList([Block3D(dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_head_dim=None)
    for i in range (self.depth)])

    def forward(self, x, y):
        for blk in self.block:
            x = blk(x, y)
        return x