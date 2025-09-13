#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.make_data_loader import make_data_loaders_base
from utils_clip import load_config_file
from utils_clip.simple_tokenizer import SimpleTokenizer
from CLIP.model import CLIP
from torch.optim.lr_scheduler import MultiStepLR
from models_ours.sanet import BlendScheduler
import utils

import models_ours
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
import numpy as np
import math
import time
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from utils_clip import set_seed
from datasets.data_augment import augment_patch_histmatch_then_augment, visualize_patch_triplet
import loss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# This is single GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

checkpoint_path   = "./saved_checkpoints/checkpoint_42_42000.pt"
MODEL_CONFIG_PATH = "./CLIP/model_config.yaml"
model_config = load_config_file(MODEL_CONFIG_PATH)

tokenizer = SimpleTokenizer()
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# This is single GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'], map_location='cpu')

        model_G = models_ours.make({
            'name': sv_file['model_G']['name'],
            'args': sv_file['model_G']['args']
        }, load_sd=False).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=False
        )

        epoch_start = sv_file['epoch'] + 1

        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler_G.step()

    else:
        model_G = models_ours.make(config['model_G']).cuda()
        optimizer_G = utils.make_optimizer_G(model_G.parameters(), config['optimizer_G'])
        epoch_start = 1

        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

    log('model_G: #params={}'.format(utils.compute_num_params(model_G, text=True)))
    return model_G, optimizer_G, epoch_start, lr_scheduler_G

def train(device, train_loader, model_G,optimizer_G, save_path,epoch,sobel=False, sv_file=False, use_tqdm=True):
    model_G.train() 
    if config['use_sanet']: 
        scheduler_blend = BlendScheduler(model_G.fusion_sanet, start=0.05, end=0.8, total_epochs=500, mode="cosine")

    loss_structure = utils.Averager()
    loss_synreal_total = utils.Averager()
    loss_clip_text = utils.Averager()
    loss_clip_img = utils.Averager()
    count = 0 # For drawing
    count1 = 0 # For drawing

    output_dir = save_path + "/vis_compare"
    txt_save_path = os.path.join(save_path, "composed_prompts.txt")
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        t0 = time.perf_counter()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device).float()
            else:
                # print('batch[{}]: {}'.format(k, v), flush=True)
                batch[k] = v

        shift_patches = batch['shift_img_patches']
        aug_patches = batch['aug_img_patches']
        seq_src = batch['seq_src'].to(device)       # [1, 1, 768]
        seq_tgt = batch['fake_seq_tgt'].to(device)  # [1, 1, 768]
        full_ref_img = batch['CLIP_full_refer'].to(device)  # [8,96,96,96]

        idx_patch = np.random.randint(len(shift_patches))

        # whole image based augmentation
        shift_patch_batch = shift_patches[idx_patch].to(device).float()
        aug_patch_batch = aug_patches[idx_patch].to(device).float()

        B = shift_patch_batch.size(0)
        # print('B:', B)

        for i in range(B):
            shift_patch = shift_patch_batch[i]  # [1, D, H, W]
            aug_patch = aug_patch_batch[i]      # [1, D, H, W

            if seq_src.shape[0] != 1:
                seq_src_single = seq_src[i].unsqueeze(0)  # shape: [1, 1, 768]
            else:
                seq_src_single = seq_src
                
            if seq_tgt.shape[0] != 1:
                seq_tgt_single = seq_tgt[i].unsqueeze(0) # shape: [1, 1, 768]
            else:
                seq_tgt_single = seq_tgt
                
            # ==== forward ====
            aug_patch = aug_patch.to(device)
            shift_patch = shift_patch.to(device)
            seq_tgt_single = seq_tgt_single.to(device)
            seq_src_single = seq_src_single.to(device)

            full_ref_img_embedding = model.encode_image(full_ref_img[i].unsqueeze(0).unsqueeze(0).to(device))  # torch.Size([1, 768])
           
            if config['use_sanet']:
                pred_patch, *_ = model_G(aug_patch, shift_patch, seq_tgt_single, seq_src_single, full_ref_img_embedding, use_adapter=config['use_adapter'],cond_mode=config['cond_mode'], use_sanet = config['use_sanet'])   # shape: [1, D, H, W] TODO
            else:
                pred_patch, *_ = model_G(aug_patch, shift_patch, seq_tgt_single, seq_src_single)

            shift_patch.requires_grad_()
            pred_patch = pred_patch.squeeze(0)  # shape: [D, H, W]
            # CLIP img feature loss
            pred_patch_img_embedding = model.encode_image(utils.preprocess_for_clip(pred_patch))  # torch.Size([1, 768])
            shift_patch_img_embedding = model.encode_image(utils.preprocess_for_clip(shift_patch))  # torch.Size([1, 768])
            CLIP_img_text_loss = 1 - F.cosine_similarity(pred_patch_img_embedding, seq_src_single[..., :768], dim=-1).mean()
            CLIP_img_loss = 1 - F.cosine_similarity(pred_patch_img_embedding, shift_patch_img_embedding, dim=-1).mean()
            # print(f"CLIP_img_text_loss: {CLIP_img_text_loss.item():.4f}, CLIP_img_loss: {CLIP_img_loss.item():.4f}", flush=True)

            L1loss = F.l1_loss(pred_patch, shift_patch.squeeze(0))

            struc, _ = loss.structural_loss(pred_patch.unsqueeze(0).unsqueeze(0), shift_patch.unsqueeze(0))
            # print("struc:",struc.requires_grad)
            total_loss = 5 * L1loss + 2 * struc + CLIP_img_text_loss + CLIP_img_loss # 更关注L1整体的风格
            sub_id = str(batch['sub_id'][i]).replace('/', '_').replace('"', '').replace("'", "")
            print(f"Epoch {epoch}, Batch {batch_idx}, Sub{sub_id}, L1loss: {L1loss.item():.4f}, structure loss:{struc.item():.4f},\
                  CLIP_text: {CLIP_img_text_loss.item():.4f}, CLIP_img: {CLIP_img_loss.item():.4f}", flush=True)

            optimizer_G.zero_grad()
           
            # 检查loss是否有效
            if total_loss is None or torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print('pred_patch:', pred_patch.min(), pred_patch.max())
                print('shift_patch:', shift_patch.min(), shift_patch.max())
                print('aug_patch:', aug_patch.min(), aug_patch.max())
                raise ValueError("Invalid total_loss value encountered.")

            total_loss.backward()
            has_bad_grad = False
            for n, p in model_G.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("🚨 bad grad in", n, p.grad.min().item(), p.grad.max().item())
                    has_bad_grad = True
            if has_bad_grad:
                optimizer_G.zero_grad()
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=5.0)
            if not torch.isfinite(torch.tensor(total_norm)):
                print("🚨 NaN/Inf grad norm")

                continue

            optimizer_G.step()

            # ==== Logging ====
            loss_synreal_total.add(L1loss.item())
            loss_structure.add(struc.item())
            loss_clip_text.add(CLIP_img_text_loss.item())
            loss_clip_img.add(CLIP_img_loss.item())

            if sub_id in map(str, range(1, 547)) and count < 10:  
                src_slices = utils.get_mid_slices(shift_patch.unsqueeze(0))    
                aug_patch_slices = utils.get_mid_slices(aug_patch.unsqueeze(0))
                synfake_slices = utils.get_mid_slices(pred_patch.unsqueeze(0).unsqueeze(0))

                utils.save_4img(
                    aug_patch_slices,
                    src_slices,
                    synfake_slices,
                    out_path=os.path.join(output_dir, f"compare_with_diff_epoch{epoch}_idx{sub_id}.jpg"),
                    title=f"Epoch {epoch} - Sample {sub_id}"
                )
                count += 1
                
            elif count1 < 10:
                src_slices = utils.get_mid_slices(shift_patch.unsqueeze(0)) 
                aug_patch_slices = utils.get_mid_slices(aug_patch.unsqueeze(0))
                synfake_slices = utils.get_mid_slices(pred_patch.unsqueeze(0).unsqueeze(0))

                utils.save_4img(
                    aug_patch_slices,
                    src_slices,
                    synfake_slices,
                    out_path=os.path.join(output_dir, f"compare_with_diff_epoch{epoch}_idx{sub_id}.jpg"),
                    title=f"Epoch {epoch} - Sample {sub_id}"
                )
                count1 += 1


        # tensorboard logging
        writer.add_scalar('loss_synreal_total', loss_synreal_total.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('loss_structure', loss_structure.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('loss_clip_text', loss_clip_text.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('loss_clip_img', loss_clip_img.item(), epoch * len(train_loader) + batch_idx)
    
    if config['use_sanet']:
        scheduler_blend.step_epoch(epoch)
        print(f"epoch {epoch}, blend={float(model_G.fusion_sanet.blend):.3f}")

    return loss_synreal_total.item(), loss_structure.item(), loss_clip_text.item(), loss_clip_img.item()

def main(config_, save_path):
    global config, log
    config = config_
    
    log = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    sobel=config['train_dataset']['dataset']['args'].get('sobel', False)
    # refer_style = config['refer_style']
    bset_psnr = 0
    best_ssim = 0

    train_loader = make_data_loaders_base(config['train_dataset'], 'train', log, ddp=False, num_workers=4)
    val_loader = make_data_loaders_base(config['val_dataset'], 'val', log, ddp=False, num_workers=4)

    model_G, optimizer_G, epoch_start, lr_scheduler_model = prepare_training()

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):

        t_epoch_start = timer.t()
        optimizer_G.param_groups[0]['lr'] = 0.00001
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        log_info.append('lr_G={:.6f}'.format(optimizer_G.param_groups[0]['lr']))


        if lr_scheduler_model is not None:
            lr_scheduler_model.step()

        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G.state_dict()

        optimizer_G_spec = config['optimizer_G']
        optimizer_G_spec['sd_G'] = optimizer_G.state_dict()

        sv_file = {
            'model_G': model_G_spec,
            'optimizer_G': optimizer_G_spec,
            'epoch': epoch
        }

        train_loss = train(device, train_loader, model_G, optimizer_G, save_path, epoch, sobel, sv_file)

        print('train_loss:', train_loss, flush=True)
        log_info.append('loss={:.4f}'.format(sum(train_loss[:4])))
        log_info.append('L1_loss={:.4f}'.format(train_loss[0]))
        log_info.append('loss_structure={:.4f}'.format(train_loss[1]))
        log_info.append('loss_clip_text={:.4f}'.format(train_loss[2]))
        log_info.append('loss_clip_img={:.4f}'.format(train_loss[3]))

        torch.save(sv_file, os.path.join(save_path, 'epoch-test.pth'))
        # exit()
        if (epoch_save is not None) and (epoch % epoch_save == 0) and epoch > 0:
            torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
        # if (epoch_val is not None): # debug
            # print('sobel:',sobel)
            psnr, ssim = utils.eval_psnr(device, val_loader, model_G, save_path, epoch, CLIP_model=model)
                
            log_info.append('psnr={:.4f}'.format(psnr))
            log_info.append('ssim={:.4f}'.format(ssim))
            

            if psnr > bset_psnr:
                bset_psnr = psnr
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-psnr.pth'))
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-ssim.pth'))

            model_G.train()

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_lccd_sr_model_G_only_tgt.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--path', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = args.name
    save_path1 = args.path
    if save_name is None and save_path1 is None:
        save_path = os.path.join(
            './save/',
            f"1-{timestamp}"
        )
    elif save_name is not None and save_path1 is None:
        save_path = os.path.join(
            './save/',
            f"{save_name}-{timestamp}"
        )
    elif save_name is None and save_path1 is not None:
        save_path = os.path.join(
            save_path1,
            f"2-{timestamp}"
        )
    else:
        save_path = os.path.join(
            save_path1,
            f"{save_name}-{timestamp}"
        )
    os.makedirs(save_path, exist_ok=True)

    global writer
    writer = SummaryWriter(log_dir=save_path)

    main(config, save_path)