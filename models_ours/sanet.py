import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BlendScheduler:
    def __init__(self, module_with_blend, start=0.2, end=1.0, 
                 total_epochs=None, total_steps=None, mode='linear'):
        assert (total_epochs is None) ^ (total_steps is None), \
        self.m = module_with_blend
        self.start = float(start)
        self.end = float(end)
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.mode = mode
        # 初始化到 start
        with torch.no_grad():
            self._set_val(self.start)

    def _curve(self, x: float) -> float:
        """x∈[0,1] -> y∈[0,1]"""
        x = max(0.0, min(1.0, x))
        if self.mode == 'linear':
            return x
        elif self.mode == 'cosine': 
            return 0.5 * (1 - math.cos(math.pi * x))
        elif self.mode == 'exp':     
            k = 5.0                 
            return (math.exp(k*x) - 1) / (math.exp(k) - 1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _set_val(self, v: float):
        v = float(max(min(v, self.end), min(self.start, self.end))) if self.end >= self.start \
            else float(max(min(v, self.start), self.end))
        device = next(self.m.parameters()).device if any(p.requires_grad for p in self.m.parameters()) else 'cpu'
        t = torch.tensor(v, device=device, dtype=torch.float32)
        with torch.no_grad():
            if isinstance(self.m.blend, torch.nn.Parameter):
                self.m.blend.data.copy_(t)
            else:
                self.m.blend.copy_(t)

    def step_epoch(self, epoch_idx: int):
        assert self.total_epochs is not None
        prog = (epoch_idx + 1) / float(self.total_epochs)
        v = self.start + (self.end - self.start) * self._curve(prog)
        self._set_val(v)

    def step_iter(self, global_step: int):
        assert self.total_steps is not None
        prog = min(1.0, global_step / float(self.total_steps))
        v = self.start + (self.end - self.start) * self._curve(prog)
        self._set_val(v)

def calc_mean_std_2d(feat, eps=1e-6):
    assert x.dim() == 4, f"calc_mean_std_2d_safe expects 4D [B,C,H,W], got {tuple(x.shape)}"
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    mean = x.mean(dim=(2, 3), keepdim=True)                                      # [B,C,1,1]
    var  = x.var (dim=(2, 3), keepdim=True, unbiased=False)                      # [B,C,1,1]
    std  = (var + eps).sqrt().clamp_(min=eps, max=1e2)                           # [B,C,1,1]
    return mean, std


def mean_variance_norm(feat, eps=1e-6):
    mean, std = calc_mean_std_2d(feat, eps)
    norm_feat = (feat - mean) / std
    return norm_feat

def _chk(name, t):
    if not torch.isfinite(t).all():
        tmin, tmax = t.min().item(), t.max().item()
        print(f"[NaN@{name}] min={tmin:.3e} max={tmax:.3e} shape={tuple(t.shape)}")
        raise FloatingPointError(f"NaN/Inf at {name}")
# -------------------------
# Positional Mapper
# -------------------------
class Mapper2D(nn.Module):
    def __init__(self, channels=512, H=16, W=16):
        super().__init__()
        self.H, self.W = H, W
        self.channels = channels

        self.rel_h = nn.Parameter(torch.zeros(1, channels, H, 1))
        self.rel_w = nn.Parameter(torch.zeros(1, channels, 1, W))

        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, style_vec):
        B, C = style_vec.shape
        # style_vec: [B,C]
        _chk("Mapper.in", style_vec)
        x = style_vec.view(B, C, 1, 1).expand(B, C, self.H, self.W)  # B,C,H,W
        _chk("Mapper.x", x)

        q = self.q(x).view(B, C, -1) ; k = self.k(x).view(B, C, -1) ; v = self.v(x).view(B, C, -1)
        _chk("Mapper.q", q); _chk("Mapper.k", k); _chk("Mapper.v", v)

        q = F.normalize(q, dim=1); k = F.normalize(k, dim=1)
        _chk("Mapper.q_norm", q); _chk("Mapper.k_norm", k)

        energy = torch.bmm(q.transpose(1,2), k) / (C ** 0.5)
        energy = energy - energy.max(dim=-1, keepdim=True).values
        _chk("Mapper.energy", energy)

        rel = (self.rel_h + self.rel_w).view(1, C, -1)
        pos = torch.bmm(q.transpose(1,2), rel.repeat(B,1,1))
        _chk("Mapper.pos", pos)

        energy = energy + pos
        energy = energy - energy.max(dim=-1, keepdim=True).values
        attn = self.softmax(energy)
        _chk("Mapper.attn", attn)

        out = torch.bmm(v, attn.transpose(1,2)).view(B, C, self.H, self.W)
        out = out + x
        _chk("Mapper.out", out)
        return out


class C_Cross_Attn_Spatial(nn.Module):
    """
    复用 cross_att.py 的写法，但输出保持空间分辨率：
      q = Conv1x1(Fc)  -> [B, heads, Nc, head_dim]
      k,v = Conv1x1(Fs) -> [B, heads, Ns, head_dim]
      out shape = [B, C, Hc, Wc]
    """
    def __init__(self, dim, num_heads=12, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = (attn_head_dim if attn_head_dim is not None else dim // num_heads)
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Fc, Fs):
        B, C, Hc, Wc = Fc.shape
        Nc = Hc * Wc
        _, C_s, Hs, Ws = Fs.shape
        assert C_s == C, "通道数不一致：Fc和Fs必须同通道"

        q = self.q(Fc).reshape(B, self.num_heads, C // self.num_heads, Nc).permute(0, 1, 3, 2)    # [B,h,Nc,dh]

        kv = self.kv(Fs).reshape(B, 2, self.num_heads, C // self.num_heads, Hs * Ws).permute(0, 1, 2, 4, 3)
        k, v = kv[:, 0], kv[:, 1]      # [B,h,Ns,dh], [B,h,Ns,dh]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # [B,h,Nc,Ns]
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                                    # [B,h,Nc,dh]
        out = out.permute(0, 1, 3, 2).reshape(B, C, Hc, Wc) # [B,C,Hc,Wc]
        out = self.proj(out)
        return out



# -------------------------
# Polynomial Attention
# -------------------------
class PolyAttn2D(nn.Module):
    def __init__(self, channels: int, R: int = 2, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            C_Cross_Attn_Spatial(dim=channels, num_heads=num_heads) for _ in range(R)
        ])
        self.scale_poly = 0.5  

    def forward(self, Fc: torch.Tensor, Fs: torch.Tensor):
        B, C, H, W = Fc.shape
        assert Fs.shape[0] == B and Fs.shape[1] == C, f"Fs shape mismatch: {Fs.shape} vs {Fc.shape}"

        Fc_n = (Fc - Fc.mean(dim=(2,3), keepdim=True)) / (
                Fc.var(dim=(2,3), keepdim=True, unbiased=False).add(1e-6).sqrt())

        mean_s = Fs.mean(dim=(2,3), keepdim=True)
        std_s  = Fs.var (dim=(2,3), keepdim=True, unbiased=False).add(1e-6).sqrt()
        Fs_n   = torch.tanh((Fs - mean_s) / std_s)   

        out = Fc_n
        cur = Fs_n
        for blk in self.blocks:
            out = out + blk(Fc_n, cur)
            cur = torch.tanh(cur * Fs_n * self.scale_poly)

        out = ((out - out.mean(dim=(2,3), keepdim=True)) /
            (out.var(dim=(2,3), keepdim=True, unbiased=False).add(1e-6).sqrt()))
        out = out * std_s + mean_s
        return out



# -------------------------
# SANet 2D
# -------------------------
class SANet2D(nn.Module):
    def __init__(self, channels=512, H=16, W=16, R=1, clip_dim=768, blend=1.0):
        super().__init__()
        self.mapper = Mapper2D(channels, H, W)   
        self.poly   = PolyAttn2D(channels, R)
        self.proj   = nn.Conv2d(channels, channels, 1)
        self.txt_proj = nn.Linear(clip_dim, channels)  
        self.low_hw = (H, W)
        # self.blend = blend  
        self.register_buffer('blend', torch.tensor(0.2))  


    @staticmethod
    def _pool_tokens(y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 2:
            return y
        elif y.dim() == 3:
            return y.mean(dim=1)
        else:
            raise ValueError(f"cond_tokens shape must be [B,768] or [B,L,768], got {tuple(y.shape)}")

    def forward(self, feat_src: torch.Tensor, cond_tokens_768: torch.Tensor):
        """
        feat_src:         [B, C, H, W]  (例如 [B, 768, 128, 128])
        cond_tokens_768:  [B, 768] 或 [B, L, 768]
        """
        B, C, H, W = feat_src.shape
        h_lr, w_lr = self.low_hw

        # 2) tokens -> [B,768] -> [B,C]
        y_768 = self._pool_tokens(cond_tokens_768)
        style_vec_c = self.txt_proj(y_768)

        Fc_lr = F.adaptive_avg_pool2d(feat_src, output_size=self.low_hw)

        # 3) mapper
        Fs_lr = self.mapper(style_vec_c)

        # 4) poly
        Fcs_lr = self.poly(Fc_lr, Fs_lr)

        Fcs_lr = self.proj(Fcs_lr)
        
        Fcs = F.interpolate(Fcs_lr, size=(H, W), mode='bilinear', align_corners=False)
        
        out = feat_src + self.blend * Fcs
        return out