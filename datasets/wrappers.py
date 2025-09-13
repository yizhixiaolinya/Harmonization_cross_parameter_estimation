import random
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from datasets import register
import numpy as np
from scipy import ndimage as nd
import utils
import re
import torch
import matplotlib.pyplot as plt
from datasets.data_augment import augment_patch_histmatch_then_augment,augment_no_tissue, visualize_patch_triplet

def crop_nonzero(volume, threshold=1e-6):
    """裁剪非0区域的tight bounding box"""
    non_zero = np.nonzero(volume > threshold)
    if len(non_zero[0]) == 0:  
        return volume
    min_idx = np.min(non_zero, axis=1)
    max_idx = np.max(non_zero, axis=1) + 1
    return volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]

# And this is version of 2.5d
@register('patch_and_whole') # this is for return the all img patches list, not a single patch
class Patches_whole(Dataset):

    def __init__(self, dataset, scale_min=1, scale_max=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # here, if we use tissue, "sobel" means tissue
        patch_src_hr, patch_src_hr_sobel, patch_tgt_hr_sobel, seq_src, fake_seq_tgt, scanner_seq_src, scanner_fake_seq_tgt, text_src, text_fake, sub_id, \
        scanner_model_part_src, other_vendor_img, mode = self.dataset[idx]
        # print('scanner_model_part_src',scanner_model_part_src,flush=True)
        patch_src_hr = utils.percentile_clip(patch_src_hr) #去除极端值
        other_vendor_img = utils.percentile_clip(other_vendor_img) if other_vendor_img is not None else None

        non_zero = np.nonzero(patch_src_hr)
        min_indice = np.min(non_zero, axis=1)
        max_indice = np.max(non_zero, axis=1)
        patch_src_hr = patch_src_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        if patch_src_hr_sobel is not None:
            patch_src_hr_sobel = patch_src_hr_sobel[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        if patch_tgt_hr_sobel is not None:
            patch_tgt_hr_sobel = patch_tgt_hr_sobel[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]

        # 处理other_vendor_img, 取它的非零块，然后reshape到src的大小
        non_zero_other_vendor = np.nonzero(other_vendor_img)
        min_indice_other_vendor = np.min(non_zero_other_vendor, axis=1)
        max_indice_other_vendor = np.max(non_zero_other_vendor, axis=1)
        # 这里我只需要它的非0图像，然后reshape到src的大小
        other_vendor_img = other_vendor_img[min_indice_other_vendor[0]:max_indice_other_vendor[0]+1,
                                            min_indice_other_vendor[1]:max_indice_other_vendor[1]+1,
                                            min_indice_other_vendor[2]:max_indice_other_vendor[2]+1]
        if mode == 'train':
            CLIP_other_vendor_img = patch_src_hr.copy() # TODO
        else:
            CLIP_other_vendor_img = other_vendor_img.copy() 
        size = 128 # 这里不做超分，lr-hr大小一致
        h_size = 8 # TODO
        use_tissue = False # TODO
        s=1

        # This can make tissue map continuous, it's not proper
        # But the input images should be same shape
        if patch_src_hr.shape != (size, size, size):
            zoom_factors = [size / s for s in patch_src_hr.shape]
            patch_src_hr = nd.zoom(patch_src_hr, zoom=zoom_factors, order=3)
        if patch_src_hr_sobel is not None and patch_src_hr_sobel.shape != (size, size, size):
            zoom_factors = [size / s for s in patch_src_hr_sobel.shape]
            patch_src_hr_sobel = nd.zoom(patch_src_hr_sobel, zoom=zoom_factors, order=0)
        if other_vendor_img.shape != (size, size, size):
            zoom_factors = [size / s for s in other_vendor_img.shape]
            other_vendor_img = nd.zoom(other_vendor_img, zoom=zoom_factors, order=3)
        if patch_tgt_hr_sobel is not None and patch_tgt_hr_sobel.shape != (size, size, size):
            zoom_factors = [size / s for s in patch_tgt_hr_sobel.shape]
            patch_tgt_hr_sobel = nd.zoom(patch_tgt_hr_sobel, zoom=zoom_factors, order=0)
        # 压缩一个ref到96，方便后续进CLIP
        if CLIP_other_vendor_img.shape != (96, 96, 96):
            zoom_factors = [96 / s for s in CLIP_other_vendor_img.shape]
            CLIP_other_vendor_img = nd.zoom(CLIP_other_vendor_img, zoom=zoom_factors, order=3)

        # TODO: 这里做whole image的数据增强，not patch based
        if use_tissue:
            shift_img, aug_img = augment_patch_histmatch_then_augment(patch_src_hr, other_vendor_img, patch_src_hr_sobel, patch_tgt_hr_sobel)
            # visualize_patch_triplet(patch_src_hr, shift_img, aug_img)
            # exit()
        else:
            print('no tissue augmentation')
            shift_img, aug_img = augment_no_tissue(patch_src_hr, other_vendor_img)

        patches,patch_idx = utils.slice_volume(patch_src_hr, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25)
        if patch_src_hr_sobel is not None and patch_tgt_hr_sobel is not None:
            patches_sobel, patch_sobel_idx = utils.slice_volume(patch_src_hr_sobel, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25) if patch_src_hr_sobel is not None else (None, None)
            patches_refer_sobel, patch_refer_sobel_idx = utils.slice_volume(patch_tgt_hr_sobel, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25) if patch_tgt_hr_sobel is not None else (None, None)

        patches_refer, patch_refer_idx = utils.slice_volume(other_vendor_img, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25) if other_vendor_img is not None else (None, None)
        # 数据增强切块
        shift_img_patches, shift_img_patch_idx = utils.slice_volume(shift_img, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25)
        aug_img_patches, aug_img_patch_idx = utils.slice_volume(aug_img, h_size=h_size, patch_size=(size, size), overlap_ratio=0.25)
        
        # print('patches[0] shape:', patches[0].shape) # patches[0] shape: torch.Size([1, 32, 32, 32])
        # print('len(patches):', len(patches), flush=True) # 120，不一定长度
        coord_hr = utils.make_coord([h_size,size,size], flatten=True) # (216000,3)
        # print('seq_src:', seq_src.shape, 'fake_seq_tgt:', fake_seq_tgt.shape, 'scanner_seq_src:', scanner_seq_src.shape, 'scanner_fake_seq_tgt:', scanner_fake_seq_tgt.shape, flush=True)

        # print('min and max of sobel patch:', patches_sobel[1].min(), patches_sobel[1].max(),flush=True)
        # min and max of sobel patch: tensor(0.) tensor(3.)

        return {
            'full_src': patch_src_hr, 
            'full_sobel': patch_src_hr_sobel, # tissue
            'CLIP_full_refer': CLIP_other_vendor_img, 
            'patches': patches,
            'patch_idx': patch_idx,
            'patches_sobel': patches_sobel,
            'patch_sobel_idx': patch_sobel_idx, # same as patch_idx
            'patches_refer': patches_refer, # whole fake style img切块
            'patch_refer_idx': patch_refer_idx, # whole fake style img切块的索引
            'patches_refer_sobel': patches_refer_sobel, 
            'shift_img_patches': shift_img_patches,
            'shift_img_patch_idx': shift_img_patch_idx,
            'aug_img_patches': aug_img_patches,
            'aug_img_patch_idx': aug_img_patch_idx,
            'patch_refer_sobel_idx': patch_refer_sobel_idx, 
            'coord_hr': coord_hr, # used for merge patches
            'seq_src': seq_src, # 这里是最开始输入生成模型infer的完整prompt
            'fake_seq_tgt':fake_seq_tgt, 
            'scanner_seq_src':scanner_seq_src,
            'scanner_fake_seq_tgt':scanner_fake_seq_tgt,
            'text_src': text_src, # 这里是最开始输入生成模型infer的完整prompt
            'text_fake': text_fake, # 这里是最开始输入生成模型refer的完整prompt
            'sub_id': sub_id,
            'scanner_model_part_src':scanner_model_part_src, # real img scanner
            'fake_style': other_vendor_img
        }
    

# For the case where we want to return all img patches list, not a single patch
@register('patches_dataset')
class PatchDataset(Dataset):
    def __init__(self, dataset, scale_min, scale_max, augment, sample_q, **kwargs):
        self.dataset = Patches_whole(self, dataset, scale_min, scale_max, augment, sample_q, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
