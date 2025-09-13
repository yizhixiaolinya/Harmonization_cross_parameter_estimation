import torch
from torch.utils.data import Dataset
from datasets import register
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np

from utils_clip import load_config_file
import SimpleITK as sitk
import re
import random
import utils

# global_model_holder.py
import torch
from utils_clip.simple_tokenizer import SimpleTokenizer
from utils_clip import load_config_file
import matplotlib.pyplot as plt

# Pay attention to this, which register the dataset class
@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path_1, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat=1, cache='none',sobel=False):
        self.repeat = repeat
        self.cache = cache
        self.root_path_1 = root_path_1
        self.prompt_pt = prompt_pt
        self.scanner_pt = scanner_pt
        self.prompt_D1_M1 = prompt_D1_M1
        self.scanner_model_bank = scanner_model_bank
        self.sobel = sobel

        with open(self.root_path_1) as f1, open(self.prompt_pt) as f_pt, open(self.scanner_pt) as f_scanner, open(self.prompt_D1_M1) as f2:
            img_M1 = f1.readlines()
            prompt_pt = f_pt.readlines()
            scanner_pt = f_scanner.readlines()
            prompt_M1 = f2.readlines()

        self.img_M1 = img_M1
        self.prompt_pt = prompt_pt
        self.scanner_pt = scanner_pt
        self.prompt_M1 = prompt_M1

    def __len__(self):
        return len(self.img_M1) * self.repeat

    def __getitem__(self, idx):
        patch_src_hr = self.img_M1[idx % len(self.img_M1)]
        text_pt = self.prompt_pt[idx % len(self.img_M1)]
        scanner_pt = self.scanner_pt[idx % len(self.img_M1)]
        text_src = self.prompt_M1[idx % len(self.img_M1)]

        sub_id, text_src = utils.extract_useful_prompt(text_src)
        if text_src.endswith('"'):
            text_src = text_src[:-1]
        # print('text_src:',text_src) # text_src: Scanner: 3.0 Siemens; Model: Prisma; Modality: T1w; FOV(mm): (256, 256); Pixel Bandwidth(Hz/Px): 220; Slices Number: 208; Voxel size: (0.8, 0.8, 0.8); TR(ms), TE(ms), TI(ms), and FA(degree): (2500.0, 2.2, 1000.0, 8.0)
        # print('text_src:',text_src) # 估计是# text_src: Scanner: 3.0 Siemens; Model: Prisma; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging Parameters TR(ms), TE(ms), TI(ms), and FA(degree): (2500.0, 2.2, 1000.0, 8.0)

        match_src = re.search(r"(Scanner:[^;]+;\s*Model:[^;]+;\s*Modality:[^;]+;)(.*)", text_src)

        if match_src:
            scanner_model_modality_src = match_src.group(1).strip()
            param_part_src = match_src.group(2).strip()
        else:
            raise ValueError("Scanner and Model pattern not matched.")

        with open(self.scanner_model_bank, "r") as f:
            scanner_model_list = [line.strip() for line in f if line.strip()]

        scanner_model_entries = []
        for line in scanner_model_list:
            try:
                scanner_info, param_info = utils.split_scanner_param(line)
                scanner_model_entries.append((scanner_info, param_info))
            except ValueError:
                continue  # 或 raise

        current_idx = -1
        for i, (scanner_info, _) in enumerate(scanner_model_entries):
            if scanner_info == scanner_model_modality_src:
                current_idx = i
                break

        candidates = []
        for scanner_info, param_info in scanner_model_entries:
            if (scanner_info != scanner_model_modality_src) and (param_info != param_part_src):
                candidates.append((scanner_info, param_info))

        if not candidates:
            raise ValueError("❗ No valid different scanner+param combination found.")

        scanner_model_modality_tgt, param_part_tgt = random.choice(candidates)
        fake_seq_tgt = scanner_model_modality_tgt + " " + param_part_tgt
 
        tgt_idx = -1
        for i, prompt in enumerate(self.prompt_M1):
            prompt_cleaned = utils.extract_useful_prompt(prompt)[1].strip('"')
            if prompt_cleaned.strip() == fake_seq_tgt.strip():
                tgt_idx = i
                break
           
        if tgt_idx == -1:
            raise ValueError("❗ No matching prompt found for fake_seq_tgt.")

        patch_fake_tgt = self.img_M1[tgt_idx]        
        text_pt_tgt_path = self.prompt_pt[tgt_idx]
        scanner_pt_src_path = self.scanner_pt[current_idx]
        scanner_pt_tgt_path = self.scanner_pt[tgt_idx]
        text_fake = self.prompt_M1[tgt_idx]
        _, text_fake = utils.extract_useful_prompt(text_fake)

        if text_fake.endswith('"'):
            text_fake = text_fake[:-1]


        seq_src = torch.load(text_pt.strip())  # (1, 1536)
        fake_seq_tgt = torch.load(text_pt_tgt_path.strip()) 

        seq_scanner_real_embedding = torch.load(scanner_pt_src_path.strip())  # (1, 768)
        seq_scanner_fake_embedding = torch.load(scanner_pt_tgt_path.strip())  # (1, 768)

        # load img
        if patch_src_hr.strip().endswith('.npy') or patch_fake_tgt.strip().endswith('.npy'):
            img_vol_src_hr = np.load(patch_src_hr.strip()) 
            img_vol_fake_tgt = np.load(patch_fake_tgt.strip()) 
        elif patch_src_hr.strip().endswith(('.nii', '.nii.gz')) or patch_fake_tgt.strip().endswith(('.nii', '.nii.gz')):
            img_vol_src_hr = sitk.GetArrayFromImage(sitk.ReadImage(patch_src_hr.strip()))
            img_vol_fake_tgt = sitk.GetArrayFromImage(sitk.ReadImage(patch_fake_tgt.strip()))


        if self.sobel:
            img_vol_src_hr_sobel = utils.get_sobel_from_array(img_vol_src_hr)
        else:
            img_vol_src_hr_sobel = None
        
        # print('seq_src:', seq_src.shape, 'fake_seq_tgt:', fake_seq_tgt.shape, 'scanner_seq_src:', seq_scanner_real_embedding.shape, 'scanner_fake_seq_tgt:', seq_scanner_fake_embedding.shape, flush=True)

        return img_vol_src_hr, img_vol_src_hr_sobel, seq_src, fake_seq_tgt,\
               seq_scanner_real_embedding, seq_scanner_fake_embedding,\
                text_src, text_fake, sub_id,\
                scanner_model_modality_src,img_vol_fake_tgt
                # param_part_src, scanner_model_modality_tgt,
        # last_hidden_state

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat, cache, **kwargs):
        self.dataset = ImageFolder(root_path_1, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat, cache, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

class Tissue_ImageFolder(Dataset):
    def __init__(self, root_path_1, tissue_path, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat=1, repeat_ids = list(range(1, 558)), repeat_factor=1, mode = 'train',train_epoch_size=1000, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.root_path_1 = root_path_1
        self.tissue_path = tissue_path
        self.prompt_pt = prompt_pt
        self.scanner_pt = scanner_pt
        self.prompt_D1_M1 = prompt_D1_M1
        self.scanner_model_bank = scanner_model_bank

        with open(self.root_path_1) as f1:
            img_M1 = f1.readlines()
        if tissue_path is not None:
            with open(self.tissue_path) as f_tissue:
                tissue_M1 = f_tissue.readlines()
            self.tissue_M1 = tissue_M1
        with open(self.prompt_pt) as f_pt: 
            prompt_pt = f_pt.readlines()
        with open(self.scanner_pt) as f_scanner:
            scanner_pt = f_scanner.readlines()
        with open(self.prompt_D1_M1) as f2:
            prompt_M1 = f2.readlines()

        self.img_M1 = img_M1
        self.prompt_pt = prompt_pt
        self.scanner_pt = scanner_pt
        self.prompt_M1 = prompt_M1

        self.repeat_ids = set(str(i) for i in repeat_ids) if repeat_ids else set()
        # print('repeat_ids:', self.repeat_ids)
        self.repeat_factor = int(repeat_factor)
        self.mode = mode  # 'train' or 'val' or 'test'
        self.train_epoch_size = train_epoch_size 

        if mode == 'train':
            self.indices = []
            for idx in range(len(self.img_M1)):
                sid_raw, _ = utils.extract_useful_prompt(self.prompt_M1[idx])
                sid = str(sid_raw).strip().strip('"').strip("'")
                if sid in self.repeat_ids:
                    # print(f"[build indices] idx={idx}, sub_id={sid} is in repeat_ids, repeating {self.repeat_factor} times")
                    self.indices.extend([idx] * int(self.repeat_factor))
                else:
                    self.indices.append(idx)
        else:
            self.indices = list(range(len(self.img_M1)))

        num_hit = sum(
            1 for idx in range(len(self.img_M1))
            if str(utils.extract_useful_prompt(self.prompt_M1[idx])[0]).strip().strip('"').strip("'") in self.repeat_ids
        )

        with open(self.scanner_model_bank, "r") as f:
            scanner_model_list = [line.strip() for line in f if line.strip()]

        self.scanner_model_entries = []
        for line in scanner_model_list:
            try:
                scanner_info, param_info = utils.split_scanner_param(line)
                self.scanner_model_entries.append((scanner_info, param_info))
            except ValueError:
                continue

    def __len__(self):
        if self.mode == 'train':
            return self.train_epoch_size
        else:
            return len(self.indices) * self.repeat

    def __getitem__(self, idx):
        if self.mode == 'train':
            scanner_model_modality_src, param_part_src = random.choice(self.scanner_model_entries)
            real_idx = None
            for i, txt in enumerate(self.prompt_M1):
                prompt_cleaned = utils.extract_useful_prompt(txt)[1].strip('"')
                
                if scanner_model_modality_src in prompt_cleaned and param_part_src in prompt_cleaned:
                    real_idx = i
                    break
            if real_idx is None:
                raise ValueError(f"找不到与 {scanner_model_modality_src} {param_part_src} 对应的样本！")
       
            patch_src_hr = self.img_M1[real_idx]
            if self.tissue_path is not None:
                tissue_src   = self.tissue_M1[real_idx]
            text_pt      = self.prompt_pt[real_idx]
            scanner_pt_src_path = self.scanner_pt[real_idx]   
            text_src     = self.prompt_M1[real_idx] 
            sub_id, text_src = utils.extract_useful_prompt(text_src)

        else:
            real_idx = self.indices[idx % len(self.indices)]

            patch_src_hr = self.img_M1[real_idx]
            if self.tissue_path is not None:
                tissue_src = self.tissue_M1[real_idx]
            text_pt = self.prompt_pt[real_idx]
            scanner_pt_src_path = self.scanner_pt[real_idx]   # 修正：以样本真实行号为准
            text_src = self.prompt_M1[real_idx]

            sub_id, text_src = utils.extract_useful_prompt(text_src)
            # 去掉两边的引号（如果有的话）
            if text_src.endswith('"'):
                text_src = text_src[:-1]
            # print('text_src:',text_src) # text_src: Scanner: 3.0 Siemens; Model: Prisma; Modality: T1w; FOV(mm): (256, 256); Pixel Bandwidth(Hz/Px): 220; Slices Number: 208; Voxel size: (0.8, 0.8, 0.8); TR(ms), TE(ms), TI(ms), and FA(degree): (2500.0, 2.2, 1000.0, 8.0)
            # print('text_src:',text_src) # 估计是# text_src: Scanner: 3.0 Siemens; Model: Prisma; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging Parameters TR(ms), TE(ms), TI(ms), and FA(degree): (2500.0, 2.2, 1000.0, 8.0)

            # Step 1: 提取当前scanner+model和参数部分
            match_src = re.search(r"(Scanner:[^;]+;\s*Model:[^;]+;\s*Modality:[^;]+;)(.*)", text_src)
            if match_src:
                scanner_model_modality_src = match_src.group(1).strip()
                param_part_src = match_src.group(2).strip()
            else:
                raise ValueError("Scanner and Model pattern not matched.")

        # Step 4: 从 scanner_model_entries 里挑出 scanner 与 param 都不同的候选
        candidates = []
        for scanner_info, param_info in self.scanner_model_entries:
            if (scanner_info != scanner_model_modality_src) and (param_info != param_part_src):
                candidates.append((scanner_info, param_info))
        if not candidates:
            raise ValueError("❗ No valid different scanner+param combination found.")

        # Step 5: 随机选一个目标组合
        scanner_model_modality_tgt, param_part_tgt = random.choice(candidates)
        fake_seq_tgt = scanner_model_modality_tgt + " " + param_part_tgt

        # Step 6: 找到 fake_seq_tgt 在 prompt_M1 中的索引
        tgt_idx = -1
        for i, prompt in enumerate(self.prompt_M1):
            prompt_cleaned = utils.extract_useful_prompt(prompt)[1].strip('"')
            # print('prompt_cleaned:', prompt_cleaned, 'fake_seq_tgt:', fake_seq_tgt, flush=True)
            if prompt_cleaned.strip() == fake_seq_tgt.strip():
                tgt_idx = i
                break
           
        if tgt_idx == -1:
            print('fake_seq_tgt:', fake_seq_tgt, flush=True)
            raise ValueError("❗ No matching prompt found for fake_seq_tgt.")

        # print('tgt_idx:', tgt_idx, flush=True)
        patch_fake_tgt = self.img_M1[tgt_idx]        
        text_pt_tgt_path = self.prompt_pt[tgt_idx]
        scanner_pt_tgt_path = self.scanner_pt[tgt_idx]
        text_fake = self.prompt_M1[tgt_idx]
        if self.tissue_path is not None:
            tissue_tgt = self.tissue_M1[tgt_idx]
        _, text_fake = utils.extract_useful_prompt(text_fake)
        # 去掉两边的引号（如果有的话）
        if text_fake.endswith('"'):
            text_fake = text_fake[:-1]

        seq_src = torch.load(text_pt.strip())  # (1, 1536)
        fake_seq_tgt = torch.load(text_pt_tgt_path.strip()) 

        seq_scanner_real_embedding = torch.load(scanner_pt_src_path.strip())  # (1, 768)
        seq_scanner_fake_embedding = torch.load(scanner_pt_tgt_path.strip())  # (1, 768)

        img_vol_tissue_src = None
        img_vol_tissue_tgt = None
        # load img
        if patch_src_hr.strip().endswith('.npy') or patch_fake_tgt.strip().endswith('.npy') or tissue_src.strip().endswith('.npy'):
            img_vol_src_hr = np.load(patch_src_hr.strip()) 
            img_vol_fake_tgt = np.load(patch_fake_tgt.strip()) 
            if self.tissue_path is not None:
                img_vol_tissue_src = np.load(tissue_src.strip())
                img_vol_tissue_tgt = np.load(tissue_tgt.strip())
        
        elif patch_src_hr.strip().endswith('.nii.gz') or patch_fake_tgt.strip().endswith('.nii.gz'):
            patch_src_hr = sitk.ReadImage(patch_src_hr.strip())
            img_vol_src_hr = sitk.GetArrayFromImage(patch_src_hr)

            patch_fake_tgt = sitk.ReadImage(patch_fake_tgt.strip())
            img_vol_fake_tgt = sitk.GetArrayFromImage(patch_fake_tgt)

            if self.tissue_path is not None:
                tissue_src = sitk.ReadImage(tissue_src.strip())
                img_vol_tissue_src = sitk.GetArrayFromImage(tissue_src)

                tissue_tgt = sitk.ReadImage(tissue_tgt.strip())
                img_vol_tissue_tgt = sitk.GetArrayFromImage(tissue_tgt)

        # img_vol_tissue_src, img_vol_tissue_tgt
        return img_vol_src_hr, img_vol_tissue_src, img_vol_tissue_tgt, seq_src, fake_seq_tgt,\
               seq_scanner_real_embedding, seq_scanner_fake_embedding,\
                text_src, text_fake, sub_id,\
                scanner_model_modality_src,img_vol_fake_tgt, self.mode


@register('paired-tissue')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, tissue_path, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat, mode, cache, **kwargs):
        self.dataset = Tissue_ImageFolder(root_path_1, tissue_path, prompt_pt, scanner_pt, prompt_D1_M1, scanner_model_bank, repeat, mode = mode, cache=cache)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]