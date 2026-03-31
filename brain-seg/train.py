'''
新建python 3.13环境（以conda为例）
    conda create -n hw4 python=3.13 -y
    conda activate hw4

安装torch，注意cuda版本适配
    pip install torch==2.9.* torchvision --index-url https://download.pytorch.org/whl/cu128

安装其他依赖库
    pip install ipykernel==7.* matplotlib==3.* scipy==1.17.* numpy==2.3.* scikit-image==0.26.* tensorboard==2.20.* tqdm==4.* -i https://pypi.tuna.tsinghua.edu.cn/simple
'''

import os
import pathlib
import time
import json
import tqdm
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import BrainSegmentationDataset
from transform import apply_affine_transform, get_random_affine_matrix_inv
from loss import DiceLoss
# 1. 原始上采样
# from unet import UNet_Up_ConvTrans as UNet

# 2. 双线性上采样
from unet import UNet_Up_Bilinear as UNet

# 3. PixelShuffle上采样
# from unet import UNet_Up_PixelShuffle as UNet

# from unet import UNet_MaxPool as UNet
# from unet import UNet_StridedConv as UNet
# from unet import UNet_PixelUnshuffle as UNet
from utils import log_images, dsc, dataloader_worker_init


#    █  ▄▄▄▄█▄     ▄▀  ▄    ▀▄ █▄▀  █    ▄▄▄▄▄█▄▄▄▄▄▄  █▀▀█▀▀█▀█
#  ▀▀█▀▀ █  █    ▀▀▀█▀▀▀▀▄  ▀▀██▀▀ █▀▀█▀  ▄█▄█▄     █  ▀▀▀▀█▀▀▀▀
# ▀▀▀█▀▄▀ ▀▀   ▀▀▀█▀▀▀█▀▀▀▀ ▄▀ █▀ ▀  ▄▀   █ █ █ ▄▄▄▄█ ▀▀▀▀▀█▀▀▀▀▀▀
#  █ █▄ █▀▀▀█   ▄▀▄▄▀ ▄▀▀▄▄ ▄▄█▄▄▄ █ █    █▀ ▀█ █       █▀▀█▀▀█
# ▄▀▄█  █▄▄▄█  ▀ ▀▄▄▀▀ ▄▄    ▀▄▄▀   █     █▀▀▀█ █   ▄   █▄▄▄█▄█
# ▀  ▀▀▄▄▄▄▄▄▄  ▄▄▄▄▄▀▀     ▄▄▀ ▀▄▄▀ ▀▄▄  █▀▀▀█ ▀▄▄▄█ ▄▄█▄▄▄▄▄█▄█▄

data_root_dir = '/xuetangx/yufeng/brain-seg/kaggle_3m'
ckpt_root_dir = '/xuetangx/yufeng/brain-seg/ckpt'
# ckpt_name     = 'lishangzhe_ckpt_Up_ConvTrans'
# ckpt_name     = 'lishangzhe_ckpt_Up_PixelShuffle'
ckpt_name     = 'lishangzhe_ckpt_Up_Bilinear'
# ckpt_name     = 'lishangzhe_ckpt_maxpool'
# ckpt_name     = 'lishangzhe_ckpt_strided'
# ckpt_name     = 'lishangzhe_ckpt_pixelunshuffle'

batch_size    = 10
epochs        = 20
lr            = 1e-4
vis_freq      = 10      # 两次可视化预测结果的间隔
device        = torch.device('cuda:0')
workers       = 2


#  ▀▄  █  ▄  █   █ ▄▄▄█▄▄▄▄   █ ▄▄█▄▄█▄▄ ▄▄▄▄▄▄▄ ▄ █
#      █  █  █  █ ▄  █ ▄    ▄▄█▄ ▄█▄▄█▄  ▄▄█▄█▄▄ █ █
# ▀▀█  █  █  █ ▀▀█▀ █▀▀█▀    ▄█▄ █▄▄▄▄█    █ █   █ █
#   █ ▄█  █  █ ▄█▄▄▀▀▀▀█▀▀▀ █ █ ▀█▄▄▄▄█  ▄▀  █ ▄  ▄█
#   █▀ █  █  █   ▄▄ ▄▀ █ ▀▄   █ ▄▄▄█▄▄▄▄   ▄▄▄▄█▄▄▄▄
#    ▄▀      █ ▀▀ ▄▀ ▀▄▀  ▀   █ ▄▄▀  ▀▄▄ ▄▄▄▄▄▄█▄▄▄▄▄

if __name__ == '__main__':
    timestamp_str = time.strftime(r'%y%m%d-%H%M%S', time.localtime())
    ckpt_dir = pathlib.Path(ckpt_root_dir, f'{timestamp_str} {ckpt_name}')
    print(f'Save directory: {ckpt_dir}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 读取数据
    dataset_train = BrainSegmentationDataset(data_root_dir, is_train=True)
    dataset_valid = BrainSegmentationDataset(data_root_dir, is_train=False)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,  persistent_workers=True, pin_memory=True, num_workers=workers, worker_init_fn=dataloader_worker_init,)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size,               drop_last=False, persistent_workers=True, pin_memory=True, num_workers=workers, worker_init_fn=dataloader_worker_init,)

    # 创建模型
    unet = UNet(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels).to(device)

    dsc_loss = DiceLoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    summarywriter = SummaryWriter(pathlib.Path(ckpt_dir, 'logs'))

    # 训练
    loss_per_epoch_per_patient: list[list[float]] = []
    dsc_per_epoch_per_patient:  list[list[float]] = []

    best_validation_dsc = 0.
    for epoch in range(epochs):
        # train
        _ = unet.train()

        losses: list[float] = []
        epoch_loader_train = tqdm.tqdm(loader_train, desc=f'E#{epoch}', postfix='init')
        for i, (xs, ys_true, gs_inv) in enumerate(epoch_loader_train):
            xs      = xs     .to(device)
            ys_true = ys_true.to(device)
            gs_inv  = gs_inv .to(device)  # random affine transformation matrix

            with torch.no_grad():
                # data augmentation with random affine transformation
                xs      = apply_affine_transform(xs,      gs_inv, interpolation='bilinear', edge_fill_value=0.)
                ys_true = apply_affine_transform(ys_true, gs_inv, interpolation='nearest',  edge_fill_value=0.)

            ys_pred: torch.Tensor = unet(xs)
            loss: torch.Tensor = dsc_loss(ys_pred, ys_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_loader_train.set_postfix_str(f'L={np.mean(losses):0.4f}')

        mean_loss = np.mean(losses)
        summarywriter.add_scalar('train loss', mean_loss, epoch)

        # validation
        _ = unet.eval()

        n_slices_per_patient = np.bincount(dataset_valid.patient_slice_index[..., 0])
        i0_slice_per_patient = np.cumsum(n_slices_per_patient) - n_slices_per_patient
        
        loss_per_patient: list[float] = []
        dsc_per_patient:  list[float] = []
        for i_patient, (i0_slice, n_slices) in tqdm.tqdm(enumerate(zip(i0_slice_per_patient, n_slices_per_patient)), desc=f'  Validating'):
            patient_name: str = dataset_valid.patients[i_patient]

            patient_losses:  list[float] = []
            patient_ys_pred: list[np.ndarray] = []
            patient_ys_true: list[np.ndarray] = []
            for i_slice in range(i0_slice, i0_slice + n_slices):
                x, y_true, g_inv = dataset_valid[i_slice]
                xs      = x     [None].to(device)
                ys_true = y_true[None].to(device)

                with torch.no_grad():
                    ys_pred: torch.Tensor = unet(xs)
                    loss = dsc_loss(ys_pred, ys_true)

                    patient_losses .append(loss.item())
                    patient_ys_pred.append(ys_pred.squeeze(0).detach().cpu().numpy())
                    patient_ys_true.append(ys_true.squeeze(0).detach().cpu().numpy())

                    if (epoch % vis_freq == 0) or (epoch == epochs - 1):
                        img, = log_images(xs, ys_true, ys_pred)
                        summarywriter.add_image(f'val images/{i_patient} {patient_name}/{i_slice}', img, epoch, dataformats='HWC')

            loss_per_patient.append(float(np.mean(patient_losses)))
            dsc_per_patient .append(float(dsc(np.stack(patient_ys_pred), np.stack(patient_ys_true))))

        loss_per_epoch_per_patient.append(loss_per_patient)
        dsc_per_epoch_per_patient .append(dsc_per_patient)

        mean_loss = np.mean(loss_per_patient)
        mean_dsc  = np.mean(dsc_per_patient)
        summarywriter.add_scalar('val loss', mean_loss, epoch)
        summarywriter.add_scalar('val dsc',  mean_dsc,  epoch)
        print(f'  val loss: {mean_loss:0.4f}, val dsc: {mean_dsc:0.4f}')

        # save best model
        if mean_dsc > best_validation_dsc:
            best_validation_dsc = float(mean_dsc)
            torch.save(unet.state_dict(), pathlib.Path(ckpt_dir, 'unet.pt'))

    print(f'Best validation mean DSC: {best_validation_dsc:4f}')

    with open(pathlib.Path(ckpt_dir, 'losses_dsc_values.json'), 'wt', encoding='utf-8') as f:
        json.dump({
            'best_dsc': best_validation_dsc,
            'patient_names': dataset_valid.patients,
            'loss_per_epoch_per_patient': loss_per_epoch_per_patient,
            'dsc_per_epoch_per_patient':  dsc_per_epoch_per_patient,
        }, f, indent=2, ensure_ascii=False)


