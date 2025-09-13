import torch
import torch.nn as nn
from models_ours.pred_paras import ParamPredictor
from models_ours.linear import Linear
from models_ours.cross_att import Basic_block
from models_ours.sanet import SANet2D
import models_ours
from models_ours import register
import utils
import torch.nn.functional as F
from einops import rearrange
from torch.nn import MultiheadAttention

def _maybe_unsqueeze3(x):
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(1)  # [B, 1, D]
    return x  # [B, L, D]

class CoOpPrompt(nn.Module):
    def __init__(self, prompt_dim=768, prompt_len=4):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(prompt_len, prompt_dim) * 0.02)
        
    def forward(self, batch_size):
        return self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
    
class SimpleAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        out_dim: int = 768,
        num_layers: int = 3,        
        reduction: int = 4,         
        hidden_dims: tuple = None,
        act: str = "relu",    
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_proj_skip: bool = True  
    ):
        super().__init__()
        assert num_layers in (2, 3),
        
        # 激活函数
        if act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError("act only support 'gelu' or 'relu'")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_layernorm = use_layernorm
        LN = nn.LayerNorm

        if hidden_dims is None:
            if num_layers == 2:
                h1 = max(out_dim * 3 // 2, out_dim)  
                hidden_dims = (h1,)
            else:  # num_layers == 3
                h1 = max(out_dim * 2, out_dim)        # 768 -> 1536
                h2 = max(out_dim * 3 // 2, out_dim)   # 768 -> 1152
                hidden_dims = (h1, h2)
        else:
            if num_layers == 2:
                assert len(hidden_dims) == 1, "1"
            else:
                assert len(hidden_dims) == 2, "2"
        

        layers = []
        if num_layers == 2:
            h1 = hidden_dims[0]
            layers += [nn.Linear(in_dim, h1)]
            if use_layernorm: layers += [LN(h1)]
            layers += [self.act, self.dropout]
            layers += [nn.Linear(h1, out_dim)]
        else:
            h1, h2 = hidden_dims
            layers += [nn.Linear(in_dim, h1)]
            if use_layernorm: layers += [LN(h1)]
            layers += [self.act, self.dropout]
            layers += [nn.Linear(h1, h2)]
            if use_layernorm: layers += [LN(h2)]
            layers += [self.act, self.dropout]
            layers += [nn.Linear(h2, out_dim)]
        
        self.mlp = nn.Sequential(*layers)

        self.use_proj_skip = use_proj_skip or (in_dim != out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if self.use_proj_skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dim]，可为 [B, D] 或 [B, N, D]，nn.Linear 作用于最后一维
        return: [..., out_dim]
        """
        y = self.mlp(x)
        skip = self.proj(x) if self.use_proj_skip else x
        return skip + y

class CondToScaleShift(nn.Module):
    """将 cond_tokens(文本/图像/融合，维度768) 映射到每层通道的 gamma/beta"""
    def __init__(self, tok_dim=768, num_channels=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tok_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2*num_channels)  # [gamma, beta]
        )

    def forward(self, cond_tokens_768): 
        if cond_tokens_768.dim() == 3:
            t = cond_tokens_768.mean(dim=1)  
        else:
            t = cond_tokens_768           
        gb = self.mlp(t)                     
        gamma, beta = gb.chunk(2, dim=1)    
        return gamma, beta

class AdaIN2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.C = num_channels

    def forward(self, x, gamma, beta):  # x:[B,C,H,W], gamma/beta:[B,C]
        B, C, H, W = x.shape
        mu = x.mean(dim=(2,3), keepdim=True)
        var = x.var(dim=(2,3), keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)

        g = gamma.view(B, C, 1, 1)
        b = beta.view(B, C, 1, 1)
        return x_norm * (1.0 + g) + b 

@register('lccd')
class LCCD(nn.Module):

    def __init__(self, encoder_spec_src, encoder_spec_0, no_imnet):
        super().__init__()
        self.encoder_0 = models_ours.make(encoder_spec_src)
        self.encoder_1 = models_ours.make(encoder_spec_0)
        self.f_dim = self.encoder_0.out_dim
        self.fusion = Basic_block(dim=self.f_dim, num_heads=8)
        self.linear_0 = Linear(in_dim=1536, out_dim=self.f_dim*2, hidden_list=[self.f_dim, self.f_dim, self.f_dim])
        self.linear_1 = Linear(in_dim=1536, out_dim=self.f_dim*2, hidden_list=[self.f_dim, self.f_dim, self.f_dim])
        self.sigmoid = nn.Sigmoid()
        if no_imnet:
            self.imnet = None
        else:
            self.imnet = Linear(in_dim=self.f_dim, out_dim=4, hidden_list=[self.f_dim*2, self.f_dim*2, self.f_dim, self.f_dim, 512, 256, 128, 64])
        self.text_adapter = SimpleAdapter(dim=1536)

    def forward(self, src, tgt, prompt_src, prompt_tgt, use_adapter=False):
        #train together
        prompt_src = prompt_src.float()
        prompt_tgt = prompt_tgt.float()
        # print('prompt_src', prompt_src.shape, 'prompt_tgt', prompt_tgt.shape)
        if use_adapter:
            prompt_tgt = self.text_adapter(prompt_tgt)
            prompt_src = self.text_adapter(prompt_src)

        param_0_src = self.linear_0(prompt_src)
        param_0_src = self.sigmoid(param_0_src)
        alpha_0_src, beta_0_src = param_0_src[:, :, :self.f_dim], param_0_src[:, :, self.f_dim:]
        # print('param_0_src', param_0_src.shape, 'alpha_0_src', alpha_0_src.shape, 'beta_0_src', beta_0_src.shape)
        param_1_tgt = self.linear_1(prompt_tgt)
        param_1_tgt = self.sigmoid(param_1_tgt)
        alpha_1_tgt, beta_1_tgt = param_1_tgt[:, :, :self.f_dim], param_1_tgt[:, :, self.f_dim:]
        # print('param_1_tgt', param_1_tgt.shape, 'alpha_1_tgt', alpha_1_tgt.shape, 'beta_1_tgt', beta_1_tgt.shape)
        #src_tgt
        feat_0_src_tgt = self.encoder_0(src) # feat_0_src_tgt torch.Size([B, 1024, 64, 64])
        # print('feat_0_src_tgt', feat_0_src_tgt.shape)
        feat_tgt = self.encoder_0(tgt) # feat_tgt torch.Size([B, 1024, 64, 64])

        # # Here we add EFDM
        # feat_0_src_tgt = utils.exact_feature_distribution_matching_base(feat_0_src_tgt, feat_tgt)
        # print('feat_0_src_tgt', feat_0_src_tgt.shape, 'feat_tgt', feat_tgt.shape)
        # exit()

        content_src_tgt = (feat_0_src_tgt - beta_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1)) / alpha_0_src.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        content_src_tgt_1 = content_src_tgt * alpha_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1) + beta_1_tgt.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        pred_src_tgt = self.imnet(content_src_tgt_1.permute(0,2,3,1)).permute(0,3,1,2)
        # print('pred_src_tgt', pred_src_tgt.shape, 'content_src_tgt', content_src_tgt.shape, 'feat_0_src_tgt', feat_0_src_tgt.shape, 'content_src_tgt_1', content_src_tgt_1.shape)
        # pred_src_tgt torch.Size([1, 4, 64, 64]) content_src_tgt torch.Size([1, 1024, 64, 64]) feat_0_src_tgt torch.Size([1, 1024, 64, 64]) content_src_tgt_1 torch.Size([1, 1024, 64, 64])

        return pred_src_tgt, content_src_tgt, feat_0_src_tgt, content_src_tgt_1
