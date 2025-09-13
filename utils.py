import os
import time
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from torch.optim import SGD, Adam
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import re
from itertools import product

def eval_psnr(device, loader, model, save_path, epoch, CLIP_model):
    model.eval()
    metric_fn = calc_psnr
    metric_fn2 = structural_similarity
    SSIM_res = Averager() # SSIM
    PSNR_res = Averager() # PSNR

    count = 0
    output_dir = save_path + "/vis_compare_val"
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device).float()
            else:
                # print('batch[{}]: {}'.format(k, v), flush=True)
                batch[k] = v 

        patches = batch['patches']            # list of [1, 1, D, H, W]
        # tissue_patches = batch['patches_sobel']
        refer_patches = batch['patches_refer']
        # refer_tissue_patches = batch['patches_refer_sobel'] 
        seq_src = batch['seq_src'].to(device)       # [1, 1, 768]
        seq_tgt = batch['fake_seq_tgt'].to(device)  # [1, 1, 768]
        full_ref_img = batch['CLIP_full_refer'].to(device)  # [8,96,96,96]

        idx_patch = np.random.randint(len(patches))
        patch_batch = patches[idx_patch].to(device).float() 
        # tissue_patch_batch = tissue_patches[idx_patch].to(device).float() if tissue_patches is not None else None
        refer_patch_batch = refer_patches[idx_patch].to(device).float()
        # refer_patch_batch_tissue = refer_tissue_patches[idx_patch].to(device).float() if refer_tissue_patches is not None else None
        # print('patch_batch shape:', patch_batch.shape, 'refer_patch_batch shape:', refer_patch_batch.shape)

        B = patch_batch.size(0)

        for i in range(B):
            # ==== 去 batch 维度 ====
            patch = patch_batch[i]         # [1, D, H, W]
            # tissue_patch = tissue_patch_batch[i] if tissue_patch_batch is not None else None  # [1, D, H, W]
            refer_patch = refer_patch_batch[i]     # [1, D, H, W]
            
                
            if seq_tgt.shape[0] != 1:
                seq_tgt_single = seq_tgt[i].unsqueeze(0) # shape: [1, 1, 768]
            else:
                seq_tgt_single = seq_tgt
            if seq_src.shape[0] != 1:
                seq_src_single = seq_src[i].unsqueeze(0)
            else:
                seq_src_single = seq_src
                
            # ==== forward ====
            # aug_patch = aug_patch.to(device)
            # shift_patch = shift_patch.to(device)
            seq_tgt_single = seq_tgt_single.to(device)
            seq_src_single = seq_src_single.to(device)
            patch = patch.to(device)
            refer_patch = refer_patch.to(device)
            full_ref_img_embedding = CLIP_model.encode_image(full_ref_img[i].unsqueeze(0).unsqueeze(0).to(device))  # torch.Size([1, 768])
           
            with torch.no_grad():
                # pred_patch, *_ = model(patch, None, None, seq_tgt_single, full_ref_img_embedding)   # shape: [1, D, H, W] TODO
                pred_patch, *_ = model(patch, refer_patch, seq_src_single, seq_tgt_single)
     
            res1 = metric_fn2(pred_patch.squeeze(0).cpu().numpy(), patch[i].squeeze(0).cpu().numpy(), data_range=1.0)  
            SSIM_res.add(res1, B)  # pred_patch vs shift_patch 

            sub_id = str(batch['sub_id'][i]).replace('/', '_').replace('"', '').replace("'", "")

            src_slices = get_mid_slices(patch.unsqueeze(0))  
            ref_patch_slices = get_mid_slices(refer_patch.unsqueeze(0))
            if pred_patch.dim() == 4:
                pred_patch = pred_patch.squeeze(0)
            syn_slices = get_mid_slices(pred_patch.unsqueeze(0).unsqueeze(0))

            if count < 10:
                save_4img(
                    src_slices,
                    ref_patch_slices,
                    syn_slices,
                    out_path=os.path.join(output_dir, f"compare_with_diff_epoch{epoch}_idx{sub_id}.jpg"),
                    title=f"Epoch {epoch} - Sample {sub_id}"
                )
                count += 1

        res0 = metric_fn(pred_patch, patch.squeeze(0))
        PSNR_res.add(res0.item(), B)

    return PSNR_res.item(),SSIM_res.item()

def preprocess_for_clip(patch, target_shape=(96,96,96), device="cuda"):
    """
    patch: [D,H,W]
    return: [1,1,96,96,96]
    """

    if patch.dim() == 3:
        patch = patch.unsqueeze(0).unsqueeze(0).to(device)
    elif patch.dim() == 4:
        patch = patch.unsqueeze(1).to(device)

    patch_resized = F.interpolate(
        patch, size=target_shape, mode='trilinear', align_corners=False
    )

    return patch_resized  # [1,1,96,96,96]

def inference_and_merge(model, batch, sobel, device=None, refer_style=False, patches=None, seq_tgt=None, seq_src=None):
    print('Infering model_G and merging patches...')

    if not isinstance(device, torch.device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if patches is None: # This is more convenient for cycle part, here we first don't consider it
        patches = batch['patches']  # should be [1, 1, D, H, W]
        # patch_indices = batch['patch_idx']  
    # Add new, do the refer img EFDM
    patches_refer = batch['patches_refer']  # should be [1, 1, D, H, W]

    patch_indices = batch['patch_idx']  # List[Tuple[int, int, int]]
    # print('patch_indices[0]:', patch_indices[0], flush=True) # [tensor([0, 0, 0, 0]), tensor([4, 4, 4, 4]), tensor([0, 0, 0, 0]), tensor([64, 64, 64, 64]), tensor([0, 0, 0, 0]), tensor([64, 64, 64, 64])][tensor([0, 0, 0, 0]), tensor([4, 4, 4, 4]), tensor([0, 0, 0, 0]), tensor([64, 64, 64, 64]), tensor([0, 0, 0, 0]), tensor([64, 64, 64, 64])]
    # print('len(patch_indices[0]):', len(patch_indices[0]), flush=True) # len(patch_indices[0]): 6

    if seq_tgt is None: # We can change the prompt
        seq_tgt = batch['fake_seq_tgt'].to(device)  # torch.Size([1, 1, 768])

    if seq_src is None: # We can change the prompt, but here the src prompt is the same 
        seq_src = batch['seq_src'].to(device)  # torch.Size([1, 1, 768])
        # print('seq_src:', seq_src.shape, flush=True) 

    patches = [p for p in patches]  # → [1, 1, D, H, W]
    refer_patches = [p for p in patches_refer]  # → [1, 1, D, H, W]
    full_shape = batch['full_src'].shape  # 为4d (B,96,96,96)
    pred_volume = torch.zeros(full_shape).to(device)
    count_volume = torch.zeros(full_shape).to(device)

    model.eval() # TODO
    with torch.no_grad(): # TODO,放注释+缩进
    # for idx_patch, patch in enumerate(patches): # TODO,没有refer img
        for idx_patch, (patch, refer_patch) in enumerate(zip(patches, refer_patches)): # TODO 有refer img
            patch = patch.to(device).float()  #  torch.Size([1, 2, 32, 32, 32])
            refer_patch = refer_patch.to(device).float()
            B = patch.size(0)

            if patch.dim() == 5:
                for i in range(B):
                    patch_single = patch[i]  # shape: [C, D, H, W]
                    refer_patch_single = refer_patch[i]  # shape: [C, D, H, W]

                    if seq_src.shape[0] != 1:
                        seq_src_single = seq_src[i].unsqueeze(0)  # shape: [1, 1, 768]
                    else:
                        seq_src_single = seq_src
                        
                    if seq_tgt.shape[0] != 1:
                        seq_tgt_single = seq_tgt[i].unsqueeze(0) # shape: [1, 1, 768]
                    else:
                        seq_tgt_single = seq_tgt

                    # print('patch_single shape:', patch_single.shape, flush=True)  # torch.Size([1, 4, 64, 64])
                    # print('seq_src_single shape:', seq_src_single.shape, flush=True)
                    pred_patch, *_ = model(patch_single, refer_patch_single, seq_src_single, seq_tgt_single) # (1,4,64,64)
                    pred_patch = pred_patch.squeeze(0).to(device) # torch.Size([4, 64, 64])

                    # merge into volume
                    x_start = int(patch_indices[idx_patch][0][i])
                    x_end   = int(patch_indices[idx_patch][1][i])
                    y_start = int(patch_indices[idx_patch][2][i])
                    y_end   = int(patch_indices[idx_patch][3][i])
                    z_start = int(patch_indices[idx_patch][4][i])
                    z_end   = int(patch_indices[idx_patch][5][i])

                    pred_volume[i, x_start:x_end, y_start:y_end, z_start:z_end] += pred_patch # should be (B, D, H, W)
                    count_volume[i, x_start:x_end, y_start:y_end, z_start:z_end] += 1.0   

            elif patch.dim() == 4:
                pred_batch, *_ = model(patch, refer_patch, seq_src, seq_tgt)      # [1,1,D,H,W]
                pred_batch = pred_batch.squeeze(0).squeeze(0)  # [D, H, W]
                # merge into volume
                x_start = int(patch_indices[idx_patch][0])
                x_end   = int(patch_indices[idx_patch][1])
                y_start = int(patch_indices[idx_patch][2])
                y_end   = int(patch_indices[idx_patch][3])
                z_start = int(patch_indices[idx_patch][4])
                z_end   = int(patch_indices[idx_patch][5])

                pred_volume[1, x_start:x_end, y_start:y_end, z_start:z_end] += pred_patch
                count_volume[1, x_start:x_end, y_start:y_end, z_start:z_end] += 1.0                
            else:
                raise ValueError(f"Unexpected patch shape {patch.shape}")

    # Avoid division by zero
    count_volume[count_volume == 0] = 1.0
    pred_vol = pred_volume / count_volume # whole volume
    # print('pred_vol shape:', pred_vol.shape, flush=True)  # should be (B, D, H, W)

    # ✅ 清理临时变量
    # del pred_feature_list, refer_feature_list
    torch.cuda.empty_cache()

    return pred_vol

def pad_to_size_centered_3d(volume, target_size, constant_value=0):

    assert volume.ndim == 3, " (D, H, W)"
    
    d, h, w = volume.shape
    td, th, tw = target_size, target_size, target_size
    
    pad_d = max(td - d, 0)
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
 
    pad_front = pad_d // 2
    pad_back  = pad_d - pad_front
    pad_top   = pad_h // 2
    pad_bottom= pad_h - pad_top
    pad_left  = pad_w // 2
    pad_right = pad_w - pad_left
    
    padding = (
        (pad_front, pad_back),  
        (pad_top,   pad_bottom), 
        (pad_left,  pad_right)  
    )
    
    padded = np.pad(volume, padding,
                    mode="constant",
                    constant_values=constant_value)
    
    if padded.shape[0] > td or padded.shape[1] > th or padded.shape[2] > tw:
        start_d = (padded.shape[0] - td) // 2
        start_h = (padded.shape[1] - th) // 2
        start_w = (padded.shape[2] - tw) // 2
        padded = padded[
            start_d:start_d+td,
            start_h:start_h+th,
            start_w:start_w+tw
        ]
    
    return padded

def zoom_tensor(tensor,target_size):

    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        B, N, _ = tensor.shape
        D, H, W = int(round(N ** (1/3))), int(round(N ** (1/3))), int(round(N ** (1/3)))
    
        tensor = tensor.view(B, 1, D, H, W)  # [B, C=1, D, H, W]
    return F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)

def reshape_volume(flat_tensor):
    """
    将 (B, N, 1) 的 tensor reshape 成 (B, 1, D, H, W)，其中 N=D*H*W
    """
    B, N, _ = flat_tensor.shape
    D, H, W = int(round(N ** (1/3))), int(round(N ** (1/3))), int(round(N ** (1/3)))
    return flat_tensor.view(B, 1, D, H, W)

def compute_clip_loss(img_feat, text_feat):
    img_feat = F.normalize(img_feat, dim=-1)
    text_feat = F.normalize(text_feat.squeeze(1), dim=-1)
    B = img_feat.shape[0]

    if B == 1:
        loss = 1.0 - torch.sum(img_feat * text_feat)  # 1 - cos similarity

    else:
        logits = torch.matmul(img_feat, text_feat.T)  # [B, B]
        target = torch.arange(B, device=img_feat.device)
        loss_i2t = F.cross_entropy(logits, target)
        loss_t2i = F.cross_entropy(logits.T, target)
        loss = (loss_i2t + loss_t2i) / 2
    return loss


def get_device_key(scanner_model_part_src):
    device_match = re.search(r'Scanner:\s*([^;]+);\s*Model:\s*([^;]+);', scanner_model_part_src)
    if device_match:
        scanner = device_match.group(1).strip()
        Model = device_match.group(2).strip()
    return f"Scanner: {scanner}; Model: {Model};"

def safe_float(x):
    try:
        x = x.strip()
        return float(x) if x.lower() != 'nan' else float('nan')
    except:
        return float('nan')

def parse_param_line(line: str): 
    device_match = re.search(r'Scanner:\s*([^;]+);\s*Model:\s*([^;]+);', line)
    device_key = f"{device_match.group(1)}__{device_match.group(2)}"

    def get_float(x):
        try:
            return float(x)
        except:
            return float('nan')

    fov_match = re.search(r'FOV\(mm\):\s*\(([^,]+),\s*([^)]+)\)', line)
    pixbw     = re.search(r'Pixel Bandwidth\(Hz/Px\):\s*([^;]+);', line)
    slices    = re.search(r'Slices Number:\s*([^;]+);', line)
    voxel     = re.search(r'Voxel size:\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
    relax     = re.search(r'TR\(ms\), TE\(ms\), TI\(ms\), and FA\(degree\):\s*\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', line)

    values = [
        get_float(fov_match.group(1)), get_float(fov_match.group(2)),
        get_float(pixbw.group(1)), get_float(slices.group(1)),
        get_float(voxel.group(1)), get_float(voxel.group(2)), get_float(voxel.group(3)),
        get_float(relax.group(1)), get_float(relax.group(2)), get_float(relax.group(3)), get_float(relax.group(4))
    ]

    return device_key, values

def get_gt_raw_tensor_from_text(text_src: str, device_key: str):

    if device_key not in text_src:
        raise ValueError(f"Device key {device_key} not found in text_src: {text_src}")

    match = re.search(r"TR\(ms\), TE\(ms\), TI\(ms\), and FA\(degree\): \(([^,]+), ([^,]+), ([^,]+), ([^)]+)\)", text_src)
    if not match:
        raise ValueError(f"Failed to extract GT parameters from text_src: {text_src}")

    values = [safe_float(v) for v in match.groups()] 

    gt_tensor = torch.tensor(values, dtype=torch.float32)

    return gt_tensor

def compose_prompt(scanner_info, voxel_sizes, params_denorm):
    """
    scanner_info: str
    voxel_sizes: list or tuple, length 3, [vx, vy, vz]
    params_denorm: list or tuple, length 4, [tr, te, ti, fa]
    """
    vx, vy, vz = voxel_sizes
    tr, te, ti, fa = params_denorm
    return (f"{scanner_info} "
            f"Voxel size: ({vx:.1f}, {vy:.1f}, {vz:.1f}); "
            f"Imaging Parameter TR(ms), TE(ms), TI(ms), and FA(degree): "
            f"({tr:.1f}, {te:.1f}, {ti:.1f}, {fa:.1f})"
           )

def tokenize(texts, tokenizer, context_length=116):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, list):
        texts = [str(t) for t in texts]
    else:
        texts = [str(texts)]  
        
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def tokenize_bert(texts, tokenizer_bert, context_length=116):
    if isinstance(texts, str):
        texts = [texts]
        
    sot_token = tokenizer_bert("<|startoftext|>")["input_ids"]
    eot_token = tokenizer_bert("<|endoftext|>")["input_ids"]
    all_tokens = [sot_token + tokenizer_bert(text)["input_ids"] + eot_token for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def get_mid_slices(vol):
    """
    vol: Tensor of shape [B, C, D, H, W]
    Returns: dict of 2D mid-slices (axial, sagittal, coronal) from sample 0, channel 0
    """
    B, C, D, H, W = vol.shape
    vol_3d = vol[0, 0]  # shape [D, H, W]

    slices = {
        'axial':    vol_3d[D // 2, :, :],    
        'sagittal': vol_3d[:, H // 2, :],   
        'coronal':  vol_3d[:, :, W // 2],  
    }
    return slices

def to_numpy(t):
    return t.detach().cpu().numpy()


def save_4img(src_slices, refer_slices, synfake_slices, out_path, title=''):
    fig, axes = plt.subplots(3, 3, figsize=(12, 20))
    planes = ['axial', 'sagittal', 'coronal']

    def to_numpy(t):
        return t.detach().cpu().numpy()

    for j, plane in enumerate(planes):
        src = to_numpy(src_slices[plane])
        refer = to_numpy(refer_slices[plane]) 
        # fake = to_numpy(fake_slices[plane])
        synfake = to_numpy(synfake_slices[plane])

        axes[0, j].imshow(src, cmap='gray')
        axes[0, j].set_title(f'src_hr - {plane}')
        axes[0, j].axis('off')


        axes[1, j].imshow(refer, cmap='gray')
        axes[1, j].set_title(f'refer - {plane}')
        axes[1, j].axis('off')

        axes[2, j].imshow(synfake, cmap='gray')
        axes[2, j].set_title(f'Predicted - {plane}')
        axes[2, j].axis('off')


    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def split_scanner_param(line):
    """
    input like: line: Scanner: 3.0 GE MEDICAL SYSTEMS; Model: Signa HDxt; Modality: T1w; FOV(mm): (260.0, 260.0); Percent Phase FOV(%): 100.0; Pixel Bandwidth(Hz/Px): 244.1; Slices Number: None; Voxel size: (1.0, 1.0, 1.2); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree): (7.0, 2.8, 400.0, 11.0)
    """
    scanner_match = re.search(r'(Scanner:[^;]+;\s*Model:[^;]+;\s*Modality:[^;]+;)', line)
    if not scanner_match:
        raise ValueError(f"Line does not contain valid scanner+model info: {line}")
    
    scanner_info = scanner_match.group(1).strip()
    # print('scanner_info:', scanner_info, flush=True)

    param_match = re.search(
        r'(Voxel size: [^;]+;.*?TR\(ms\), TE\(ms\), TI\(ms\), and FA\(degree\): \([^)]+\))',
        line
    )
    if not param_match:
        raise ValueError(f"Line does not contain valid parameter info: {line}")
    param_part = param_match.group(1).strip()
    # print('param_part:', param_part, flush=True)
    # exit()
    return scanner_info, param_part

def extract_useful_prompt(text_src):
    # print('text_src:', text_src, flush=True)
    parts = text_src.split(":", 1)
    # print('parts:', parts, flush=True)
    if '.npy' in parts[0]:
        sub_id = parts[0].replace('.npy', '')
    elif '.nii.gz' in parts[0]:
        sub_id = parts[0].replace('.nii.gz', '')
    # print('sub_id:', sub_id, flush=True)
    description = parts[1].strip() if len(parts) > 1 else text_src

    description = description.strip('"')

    pattern = (r"(Scanner: [^;]+; Model: [^;]+; Modality: [^;]+;).*?"
               r"(Voxel size: \([^)]+\);).*?"
               r"(TR\(ms\), TE\(ms\), TI\(ms\), and FA\(degree\): \([^)]+\))")

    match = re.search(pattern, description)
    if match:
        scanner_model_modality = match.group(1).strip()
        voxel = match.group(2).strip()
        imaging_param = match.group(3).strip()
        return sub_id, f"{scanner_model_modality} {voxel} {imaging_param}"
    else:
        raise ValueError(f"Pattern not matched in text: {text_src}")

# For demo and training for all patches
def calculate_patch_index(target_size, patch_size, overlap_ratio=0.25):
    shape = target_size

    gap = int(patch_size[0] * (1 - overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0] - patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1] - patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2] - patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos

def patch_slicer(img_vol_0, overlap_ratio, crop_size, scale0, scale1, scale2):
    W, H, D = img_vol_0.shape
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    scan_patches = []
    patch_idx = []
    for start_pos in pos:
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1],
                         start_pos[2]:start_pos[2] + crop_size[2]]
        #print(img_0_lr_patch.shape)
        scan_patches.append(torch.tensor(img_0_lr_patch).float().unsqueeze(0))
        patch_idx.append([
            int(start_pos[0]),
            int(start_pos[0]) + crop_size[0],
            int(start_pos[1]),
            int(start_pos[1]) + crop_size[1],
            int(start_pos[2]),
            int(start_pos[2]) + crop_size[2],
        ])

    return scan_patches, patch_idx

def slice_volume(img_vol_0, h_size=4, patch_size=(64, 64), overlap_ratio=0.25):

    H, D, W = img_vol_0.shape
    crop_d, crop_w = patch_size
    scan_patches = []
    patch_idx = []

    positions = calculate_patch_index((H, D, W), (h_size, crop_d, crop_w), overlap_ratio)

    for (h0, d0, w0) in positions:
        patch = img_vol_0[
            h0: h0 + h_size,
            d0: d0 + crop_d,
            w0: w0 + crop_w
        ]
        if patch.shape != (h_size, crop_d, crop_w):
            continue  # Skip incomplete patch

        patch_tensor = torch.tensor(patch).float().unsqueeze(0)  # shape (1, h_size, crop_d, crop_w)
        scan_patches.append(patch_tensor)
        patch_idx.append([h0, h0 + h_size, d0, d0 + crop_d, w0, w0 + crop_w])

    return scan_patches, patch_idx

def merge_patches(patches, volume_shape, patch_size=(60, 60, 60), overlap=0.25):
    vol = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)

    gap = [int(patch_size[i] * (1 - overlap)) for i in range(3)]
    indices = calculate_patch_index(volume_shape, patch_size, overlap_ratio=overlap)

    for patch, idx in zip(patches, indices):
        patch = patch.squeeze()  # shape: (60, 60, 60)
        z, y, x = idx
        vol[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += patch
        weight[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1.0

    weight[weight == 0] = 1e-5
    return vol / weight

def merge_patches_3d(patches, patch_indices, volume_shape, patch_size=32):
    """
    Merge a list of 3D patches back into a full volume.

    Args:
        patches (list of tensors): list of (1, 1, D, H, W) predicted patches (must be same size)
        patch_indices (list of tuples): list of (z, y, x) coordinates for each patch's top-left corner
        volume_shape (tuple): full volume shape as (1, D, H, W)
        patch_size (int): size of each patch cube (default=32)

    Returns:
        Tensor: merged volume of shape (1, 1, D, H, W)
    """
    C = patches[0].shape[1]
    merged = torch.zeros((1, C) + volume_shape[1:], dtype=patches[0].dtype).to(patches[0].device)
    weight = torch.zeros_like(merged)

    for patch, (z, y, x) in zip(patches, patch_indices):
        merged[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += patch
        weight[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1.0

    # Avoid division by zero
    weight[weight == 0] = 1.0
    merged =merged/ weight

    return merged


def percentile_clip(input_tensor, reference_tensor=None, p_min=0.01, p_max=99.9, strictlyPositive=True):
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile
    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]
    return output_tensor

def random_selection(input_list):
    num_to_select = random.randint(1, 3)
    selected_numbers = random.sample(input_list, num_to_select)
    return selected_numbers

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    #writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer_G(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        # optimizer.load_state_dict(optimizer_spec['sd_G'])
        if 'sd_G' in optimizer_spec:
            optimizer.load_state_dict(optimizer_spec['sd_G'])
        if 'sd_P' in optimizer_spec:
            optimizer.load_state_dict(optimizer_spec['sd_P'])
        if 'sd_D' in optimizer_spec:
            optimizer.load_state_dict(optimizer_spec['sd_D'])
        if 'sd_D_A' in optimizer_spec:
            optimizer.load_state_dict(optimizer_spec['sd_D_A'])
        if 'sd_D_B' in optimizer_spec:
            optimizer.load_state_dict(optimizer_spec['sd_D_B'])
    return optimizer

def make_optimizer_P(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_P'])
    return optimizer

def make_optimizer_D(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_D'])
    return optimizer

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):#遍历每个维度
        if ranges is None:
            v0, v1 = -1, 1#将坐标归一化到-1到1之间
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)#计算每两个点之间的间隔
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])#展平成2维 第一维是点的数量 第二维是每个点的坐标维度
    return ret

def calc_psnr(sr, hr):
    diff = (sr - hr) 
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse)

def write_middle_feature(intermediate_output):
    for i in range(intermediate_output.shape[1]):
        activation = intermediate_output[0, i, :, :, :]
        plt.savefig(f'./save/layer_{i}_activation_{activation}.png')  # Save each activation as a PNG file
        plt.clf()

def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)#读取参考文件
    #img = sitk.GetImageFromArray(vol.transpose(4, 1, 2, 3, 0).squeeze())#将体积数据vol转为img
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())#将参考图像的方向信息复制到新图像中
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())#使用参考图像的间距
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())#设置图像原点
    sitk.WriteImage(img, out_path)#写入图像文件
    print('Save to:', out_path)
