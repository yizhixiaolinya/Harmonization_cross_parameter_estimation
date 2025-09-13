import numpy as np
import torch
from monai.transforms import (
    Compose, MapTransform, Randomizable, RandAffined,
    RandBiasFieldd, RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd
)
import SimpleITK as sitk

import random
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import numpy as np
from datasets import MRI_Syn
# import MRI_Syn

from typing import Dict, Sequence, Tuple
import torch.nn.functional as F
from monai.transforms import (
    Compose, MapTransform, Randomizable, RandAffined,
    RandBiasFieldd, RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd
)
from monai.config import KeysCollection

def rand_gamma_adjust(tensor: torch.Tensor, gamma_range=(0.5, 1.5), prob=0.3):
    """
    Apply random gamma correction: x → x^γ
    Args:
        tensor: [C, D, H, W]
        gamma_range: tuple of (min, max)
        prob: probability to apply the transform
    Returns:
        Transformed tensor
    """
    if random.random() < prob:
        gamma = random.uniform(*gamma_range)
        tensor = tensor.clamp(min=1e-6)  # Avoid 0^gamma
        return tensor.pow(gamma)
    else:
        return tensor


joint_tf = Compose([
    RandAffined(
        keys=["image", "syn"],
        prob=1.0,
        rotate_range=(np.pi/9, np.pi/9, np.pi/9),
        translate_range=(4, 8, 8),
        mode=("bilinear", "bilinear"), 
        padding_mode="border",
        spatial_size=None
    ),
])

def augment_patch_histmatch_then_augment(patch: torch.Tensor,
                                         ref_patch: torch.Tensor,
                                         tissue: torch.Tensor,
                                         ref_tissue: torch.Tensor):
    syn_patch = MRI_Syn.GMM_mri_t1(
        patch.squeeze(), ref_patch.squeeze(),
        tissue.squeeze(), ref_tissue.squeeze()
    )

    batch = {"image": patch, "syn": syn_patch} 

    out = joint_tf(batch) 
    original_data = out["image"]
    aug_data      = out["syn"]

    # min max, because the data is transformed to 0-255 in the syn seg
    if original_data.max() == original_data.min():
        original_data = torch.zeros_like(original_data)
    else:
        original_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
    
    if aug_data.max() == aug_data.min():
        aug_data = torch.zeros_like(aug_data)
    else:
        aug_data = (aug_data - aug_data.min()) / (aug_data.max() - aug_data.min())

    # # check nan
    # print('original data min:', original_data.min(), 'max:', original_data.max(),flush=True)
    # print('aug data min:', aug_data.min(), 'max:', aug_data.max(),flush=True)
    return original_data, aug_data



# =======ADD new for no tissue augmentation=============
class RandDownUpSampled(Randomizable, MapTransform):
    """
    Randomly downsample then upsample back to original size.
    Accepts [D,H,W] or [C,D,H,W]; returns the same ndim as input.
    """
    def __init__(self, keys: KeysCollection, prob: float = 0.5,
                 scale_range=(1.5, 3.0), mode="trilinear", align_corners=False):
        super().__init__(keys)
        self.prob = prob
        self.scale_range = scale_range
        self.mode = mode
        self.align_corners = align_corners
        self._do_transform = False
        self._factor = 1.0

    def randomize(self):
        self._do_transform = self.R.random() < self.prob
        if self._do_transform:
            self._factor = self.R.uniform(*self.scale_range)

    def _to_cdhw(self, x: torch.Tensor):
        """Return x as [C,D,H,W], and a flag whether channel existed."""
        if x.ndim == 4:        # [C,D,H,W]
            return x, True
        elif x.ndim == 3:      # [D,H,W]
            return x.unsqueeze(0), False
        else:
            raise AssertionError(f"Expect 3D or 4D, got {x.ndim}D")

    def _restore_dim(self, x: torch.Tensor, had_channel: bool):
        return x if had_channel else x.squeeze(0)

    def __call__(self, data: Dict):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for k in self.keys:
            x: torch.Tensor = d[k]
            x_c, had_c = self._to_cdhw(x)                     # -> [C,D,H,W]
            c, dd, hh, ww = x_c.shape
            # down
            new_d = max(1, int(round(dd / self._factor)))
            new_h = max(1, int(round(hh / self._factor)))
            new_w = max(1, int(round(ww / self._factor)))
            x_low = F.interpolate(
                x_c.unsqueeze(0), size=(new_d, new_h, new_w),
                mode=self.mode, align_corners=self.align_corners
            ).squeeze(0)
            # up
            x_rec = F.interpolate(
                x_low.unsqueeze(0), size=(dd, hh, ww),
                mode=self.mode, align_corners=self.align_corners
            ).squeeze(0)
            d[k] = self._restore_dim(x_rec, had_c)  
        return d


class RandGammad(Randomizable, MapTransform):
    """
    Random gamma correction on dict keys.
    Accepts [D,H,W] or [C,D,H,W]; returns the same ndim.
    """
    def __init__(self, keys: KeysCollection, prob=0.5,
                 gamma_range=(0.5, 1.5), eps=1e-6):
        super().__init__(keys)
        self.prob = prob
        self.gamma_range = gamma_range
        self.eps = eps
        self._do_transform = False
        self._gamma = 1.0

    def randomize(self):
        self._do_transform = self.R.random() < self.prob
        if self._do_transform:
            self._gamma = self.R.uniform(*self.gamma_range)

    def _to_cdhw(self, x: torch.Tensor):
        if x.ndim == 4:
            return x, True
        elif x.ndim == 3:
            return x.unsqueeze(0), False
        else:
            raise AssertionError(f"Expect 3D or 4D, got {x.ndim}D")

    def _restore_dim(self, x: torch.Tensor, had_channel: bool):
        return x if had_channel else x.squeeze(0)

    def __call__(self, data: Dict):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for k in self.keys:
            x: torch.Tensor = d[k]
            x_c, had_c = self._to_cdhw(x)
            x_c = torch.clamp(x_c, min=self.eps).pow(self._gamma)
            d[k] = self._restore_dim(x_c, had_c)
        return d



# ----------------------------
def build_aug_pipeline_joint_then_synonly(
    joint_keys=("image", "syn"),
    syn_only_key=("syn",),
):
    """
    先对 joint_keys 做“可能发生的”几何变换（若未命中prob则两路都不变）；
    然后仅对 syn_only_key 做强度/分辨率类扰动。
    """
    return Compose([
        RandAffined(
            keys=joint_keys,
            prob=0.8,  
            rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            translate_range=(2, 4, 4),
            mode=("bilinear",) * len(joint_keys),
            padding_mode="border",
            spatial_size=None,
        ),

        RandBiasFieldd(keys=syn_only_key, prob=0.7, coeff_range=(0.0, 0.6), degree=3),
        RandDownUpSampled(keys=syn_only_key, prob=0.7, scale_range=(1.5, 3.0),
                          mode="trilinear", align_corners=False),
        RandGammad(keys=syn_only_key, prob=0.7, gamma_range=(0.6, 1.6)),
        RandAdjustContrastd(keys=syn_only_key, prob=0.7, gamma=(0.7, 1.4)),
        RandScaleIntensityd(keys=syn_only_key, factors=0.2, prob=0.7),
        RandShiftIntensityd(keys=syn_only_key, offsets=0.1, prob=0.7),
    ])

def augment_no_tissue(patch: torch.Tensor, ref_patch: torch.Tensor):
    batch = {"image": patch, "syn": patch} 
    aug = build_aug_pipeline_joint_then_synonly(
        joint_keys=("image", "syn"),
        syn_only_key=("syn",),
    )
    out = aug(batch)
    original_data = out["image"]
    aug_data      = out["syn"]

    # min max, because the data is transformed to 0-255 in the syn seg
    if original_data.max() == original_data.min():
        original_data = torch.zeros_like(original_data)
    else:
        original_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
    
    if aug_data.max() == aug_data.min():
        aug_data = torch.zeros_like(aug_data)
    else:
        aug_data = (aug_data - aug_data.min()) / (aug_data.max() - aug_data.min())

    # # check nan
    # print('original data min:', original_data.min(), 'max:', original_data.max(),flush=True)
    # print('aug data min:', aug_data.min(), 'max:', aug_data.max(),flush=True)
    return original_data, aug_data


# ======functions for resampling and plotting======
def resample_to_shape(itk_image, out_shape=(96,96,96), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    original_direction = itk_image.GetDirection()
    original_origin = itk_image.GetOrigin()
    out_size = out_shape
    out_spacing = [
        original_spacing[i] * original_size[i] / out_size[i] for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled = resampler.Execute(itk_image)
    return resampled

