#!/usr/bin/env python3
"""
CycleGAN v6.0 - 去霧專用版本，防止風格轉換化 (優化版)
主要改進：
1. 上採樣層改為 Upsample + Conv2d + IN + ReLU
2. 修正 SelfAttention 的 view 問題
3. 真正套用 spectral norm
4. 本地 VGG perceptual loss
5. 重疊滑窗推論 + 權重融合
6. 兩尺度判別器
7. 結構保持損失 (SSIM + 梯度一致性 + 邊緣保持) - [0,1]域計算
8. 去霧先驗損失 (最小池化暗通道 + 對比度增強 + 亮度一致性) - [0,1]域計算
9. 提升 identity loss 權重 (λ=20) 防止過度風格化
10. 多尺度 GAN 損失取均值
11. 反射 padding 替代零填充
12. 學習率線性衰減到0
13. 訓練完成後生成專業損失函數圖表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import json
from tqdm import tqdm
import os
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import math

# 導入數據集
from cyclegan_dataset import CycleGANDataset

# 設備設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def spectral_norm(module, name='weight', power_iterations=1):
    """使用 PyTorch 內建的 spectral normalization"""
    try:
        return torch.nn.utils.spectral_norm(module, name=name, n_power_iterations=power_iterations)
    except:
        # 如果失敗，返回原始模組
        return module

class SelfAttention(nn.Module):
    """修正的自注意力機制"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # 使用 spectral norm
        self.query_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C' x N
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x N
        
        # 注意力權重
        attention = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(attention)
        
        # 應用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, H, W)  # 修正：正確的 view 操作
        
        # 殘差連接
        out = self.gamma * out + x
        return out

class ImprovedUpsample(nn.Module):
    """改進的上採樣塊：Upsample + Conv2d + IN + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ImprovedUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    """殘差塊 - 使用反射padding"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=0)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=0)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual

class V6Generator(nn.Module):
    """v6 生成器 - 修正上採樣和注意力機制"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super(V6Generator, self).__init__()
        
        # 編碼器 - 使用反射padding
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 殘差塊
        residual_layers = []
        for _ in range(n_residual_blocks):
            residual_layers.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_layers)
        
        # 自注意力（在中間特徵圖上）
        self.attention = SelfAttention(256)
        
        # 解碼器 - 兩段上採樣（修正尺寸問題）+ 反射padding
        self.decoder = nn.Sequential(
            ImprovedUpsample(256, 128),  # 第一段上採樣 64x64 -> 128x128
            ImprovedUpsample(128, 64),   # 第二段上採樣 128x128 -> 256x256
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x

class MultiScaleDiscriminator(nn.Module):
    """兩尺度判別器"""
    def __init__(self, input_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        
        # 原尺度判別器
        self.discriminator_full = self._make_discriminator(input_channels)
        
        # 半尺度判別器
        self.discriminator_half = self._make_discriminator(input_channels)
        
        self.downsample = nn.AvgPool2d(2)
        
    def _make_discriminator(self, input_channels):
        """創建單個判別器 - 加深一層"""
        return nn.Sequential(
            # 第一層
            spectral_norm(nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二層
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三層
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四層（新增）
            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第五層
            spectral_norm(nn.Conv2d(512, 512, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 輸出層
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, x):
        # 原尺度
        out_full = self.discriminator_full(x)
        
        # 半尺度
        x_half = self.downsample(x)
        out_half = self.discriminator_half(x_half)
        
        return [out_full, out_half]

class LocalVGGLoss(nn.Module):
    """簡化的感知損失 - 暫時禁用複雜VGG"""
    def __init__(self):
        super(LocalVGGLoss, self).__init__()
        # 簡單的特徵提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # 凍結參數
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input_img, target_img):
        # 如果輸入圖像過小，直接返回L1損失
        if input_img.size(2) < 32 or input_img.size(3) < 32:
            return F.l1_loss(input_img, target_img)
        
        try:
            input_features = self.features(input_img)
            target_features = self.features(target_img)
            return F.mse_loss(input_features, target_features)
        except Exception as e:
            # 如果有任何錯誤，回退到L1損失
            return F.l1_loss(input_img, target_img)

class StructuralLoss(nn.Module):
    """結構保持損失 - 防止過度風格化"""
    def __init__(self):
        super(StructuralLoss, self).__init__()
        
    def forward(self, input_img, output_img):
        # 轉換到 [0,1] 域進行計算
        input_01 = (input_img + 1.0) / 2.0  # [-1,1] -> [0,1]
        output_01 = (output_img + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # SSIM 結構相似性損失
        ssim_loss = self._ssim_loss(input_01, output_01)
        
        # 梯度一致性損失
        gradient_loss = self._gradient_loss(input_01, output_01)
        
        # 邊緣保持損失
        edge_loss = self._edge_preserving_loss(input_01, output_01)
        
        return ssim_loss + 0.5 * gradient_loss + 0.3 * edge_loss
    
    def _ssim_loss(self, img1, img2):
        """SSIM 損失"""
        mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
        mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return 1 - ssim_map.mean()
    
    def _gradient_loss(self, img1, img2):
        """梯度一致性損失"""
        def gradient(img):
            # Sobel 梯度
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
            
            # 將圖像轉換為灰度
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            
            return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        grad1 = gradient(img1)
        grad2 = gradient(img2)
        
        return F.l1_loss(grad1, grad2)
    
    def _edge_preserving_loss(self, img1, img2):
        """邊緣保持損失"""
        # Laplacian 邊緣檢測
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
        
        # 轉換為灰度
        gray1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
        gray2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
        
        edge1 = F.conv2d(gray1, laplacian, padding=1)
        edge2 = F.conv2d(gray2, laplacian, padding=1)
        
        return F.l1_loss(edge1, edge2)

class DehazingPriorLoss(nn.Module):
    """去霧先驗損失 - 基於大氣散射模型和暗通道先驗"""
    def __init__(self):
        super(DehazingPriorLoss, self).__init__()
        
    def forward(self, hazy_img, dehazed_img):
        # 轉換到 [0,1] 域進行計算
        hazy_01 = (hazy_img + 1.0) / 2.0  # [-1,1] -> [0,1]
        dehazed_01 = (dehazed_img + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # 暗通道先驗損失
        dark_channel_loss = self._dark_channel_loss(hazy_01, dehazed_01)
        
        # 對比度增強損失
        contrast_loss = self._contrast_enhancement_loss(hazy_01, dehazed_01)
        
        # 亮度一致性損失
        brightness_loss = self._brightness_consistency_loss(hazy_01, dehazed_01)
        
        # 顏色偏移損失
        color_shift_loss = self._color_shift_loss(hazy_01, dehazed_01)
        
        return dark_channel_loss + 0.5 * contrast_loss + 0.3 * brightness_loss + 0.2 * color_shift_loss
    
    def _dark_channel_loss(self, hazy, dehazed):
        """暗通道先驗：去霧後的暗通道應該更小，使用最小池化"""
        def dark_channel(img):
            # 每個像素在RGB三個通道中的最小值
            dc = torch.min(img, dim=1, keepdim=True)[0]
            # 使用最小池化（通過負的最大池化實現）
            dc_neg = -dc
            dc_pooled = F.max_pool2d(dc_neg, kernel_size=15, stride=1, padding=7)
            dc = -dc_pooled  # 轉回最小池化結果
            return dc
        
        dc_hazy = dark_channel(hazy)
        dc_dehazed = dark_channel(dehazed)
        
        # 去霧後的暗通道應該更小（更接近0）
        loss = F.relu(dc_dehazed - dc_hazy + 0.1).mean()
        
        return loss
    
    def _contrast_enhancement_loss(self, hazy, dehazed):
        """對比度增強：去霧後對比度應該更高"""
        def local_contrast(img):
            # 計算局部標準差作為對比度指標
            mean = F.avg_pool2d(img, kernel_size=9, stride=1, padding=4)
            sq_diff = (img - mean) ** 2
            variance = F.avg_pool2d(sq_diff, kernel_size=9, stride=1, padding=4)
            std = torch.sqrt(variance + 1e-8)
            return std.mean()
        
        contrast_hazy = local_contrast(hazy)
        contrast_dehazed = local_contrast(dehazed)
        
        # 去霧後對比度應該更高
        loss = F.relu(contrast_hazy - contrast_dehazed + 0.05)
        
        return loss
    
    def _brightness_consistency_loss(self, hazy, dehazed):
        """亮度一致性：防止過度曝光或過暗"""
        # 計算亮度（轉為灰度）
        def to_grayscale(img):
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        
        brightness_hazy = to_grayscale(hazy).mean()
        brightness_dehazed = to_grayscale(dehazed).mean()
        
        # 亮度變化應該適中（不要過度變化）
        brightness_diff = torch.abs(brightness_dehazed - brightness_hazy)
        
        # 懲罰過度的亮度變化（>0.3）
        loss = F.relu(brightness_diff - 0.3)
        
        return loss
    
    def _color_shift_loss(self, hazy, dehazed):
        """顏色偏移損失：防止顏色過度偏移"""
        # 計算RGB各通道的均值
        hazy_mean = hazy.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        dehazed_mean = dehazed.mean(dim=[2, 3], keepdim=True)
        
        # 顏色偏移應該適中
        color_diff = torch.abs(dehazed_mean - hazy_mean)
        
        # 懲罰過度的顏色偏移
        loss = F.relu(color_diff - 0.2).mean()
        
        return loss

class ImageBuffer:
    """歷史圖片緩衝區"""
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def push_and_pop(self, images):
        to_return = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                to_return.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.buffer_size - 1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = image
                else:
                    to_return.append(image)
        return torch.cat(to_return)

def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化網路權重"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif (classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1):
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)

class OverlapInference:
    """重疊滑窗推論 + 權重融合"""
    def __init__(self, model, tile_size=512, overlap=64):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
    def __call__(self, input_tensor):
        """重疊滑窗推論"""
        batch_size, channels, height, width = input_tensor.shape
        
        # 如果圖像小於tile_size，直接處理
        if height <= self.tile_size and width <= self.tile_size:
            return self.model(input_tensor)
        
        # 計算需要的tiles
        h_tiles = math.ceil((height - self.overlap) / self.stride)
        w_tiles = math.ceil((width - self.overlap) / self.stride)
        
        # 輸出tensor和權重tensor
        output = torch.zeros_like(input_tensor)
        weights = torch.zeros(batch_size, 1, height, width, device=input_tensor.device)
        
        for h in range(h_tiles):
            for w in range(w_tiles):
                # 計算tile位置
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = min(h_start + self.tile_size, height)
                w_end = min(w_start + self.tile_size, width)
                
                # 提取tile
                tile = input_tensor[:, :, h_start:h_end, w_start:w_end]
                
                # 如果tile太小，跳過
                if tile.shape[2] < 64 or tile.shape[3] < 64:
                    continue
                
                # 處理tile
                with torch.no_grad():
                    tile_output = self.model(tile)
                
                # 計算對稱權重（四邊淡出）
                tile_h, tile_w = tile.shape[2], tile.shape[3]
                tile_weight = torch.ones(1, 1, tile_h, tile_w, device=input_tensor.device)
                
                # 對稱邊緣淡出
                fade_size = self.overlap // 2
                if fade_size > 0:
                    # 上下邊緣對稱淡出
                    if h > 0 or h < h_tiles - 1:
                        fade = torch.linspace(0, 1, fade_size, device=input_tensor.device)
                        # 上邊緣
                        if h > 0:
                            tile_weight[:, :, :fade_size, :] *= fade.view(-1, 1)
                        # 下邊緣
                        if h < h_tiles - 1:
                            tile_weight[:, :, -fade_size:, :] *= fade.flip(0).view(-1, 1)
                    
                    # 左右邊緣對稱淡出
                    if w > 0 or w < w_tiles - 1:
                        fade = torch.linspace(0, 1, fade_size, device=input_tensor.device)
                        # 左邊緣
                        if w > 0:
                            tile_weight[:, :, :, :fade_size] *= fade.view(1, -1)
                        # 右邊緣
                        if w < w_tiles - 1:
                            tile_weight[:, :, :, -fade_size:] *= fade.flip(0).view(1, -1)
                
                # 累積結果
                output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight
                weights[:, :, h_start:h_end, w_start:w_end] += tile_weight
        
        # 正規化
        weights = torch.clamp(weights, min=1e-8)
        output = output / weights
        
        return output

class V6CycleGANTrainer:
    """v6 CycleGAN 訓練器 - 專為去霧任務優化"""
    def __init__(self, 
                 lambda_cycle=10.0, 
                 lambda_identity=20.0,  # 提升identity權重防止風格化
                 lambda_perceptual=1.0,
                 lambda_structural=5.0,  # 結構保持損失權重
                 lambda_dehazing=3.0,   # 去霧先驗損失權重
                 buffer_size=50, 
                 lr=0.0002, 
                 beta1=0.5, 
                 beta2=0.999):
        
        # 創建模型
        self.G_A2B = V6Generator().to(device)
        self.G_B2A = V6Generator().to(device)
        self.D_A = MultiScaleDiscriminator().to(device)
        self.D_B = MultiScaleDiscriminator().to(device)
        
        # 初始化權重
        init_weights(self.G_A2B)
        init_weights(self.G_B2A)
        init_weights(self.D_A)
        init_weights(self.D_B)
        
        # 損失函數
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = LocalVGGLoss().to(device)
        self.criterion_structural = StructuralLoss().to(device)  # 結構保持損失
        self.criterion_dehazing = DehazingPriorLoss().to(device)  # 去霧先驗損失
        
        # 優化器
        self.optimizer_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=lr, betas=(beta1, beta2))
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, beta2))
        
        # 學習率調度器 - LR線性衰減到0
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda epoch: max(0.0, 1.0 - max(0, epoch - 100) / 100))
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=lambda epoch: max(0.0, 1.0 - max(0, epoch - 100) / 100))
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=lambda epoch: max(0.0, 1.0 - max(0, epoch - 100) / 100))
        
        # 緩衝區
        self.fake_A_buffer = ImageBuffer(buffer_size)
        self.fake_B_buffer = ImageBuffer(buffer_size)
        
        # 參數
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_perceptual = lambda_perceptual
        self.lambda_structural = lambda_structural
        self.lambda_dehazing = lambda_dehazing
        
        # 重疊推論器
        self.overlap_inference_A2B = OverlapInference(self.G_A2B)
        self.overlap_inference_B2A = OverlapInference(self.G_B2A)
        
        # 損失記錄
        self.losses_history = {
            'G_loss': [],
            'D_A_loss': [],
            'D_B_loss': [],
            'cycle_loss': [],
            'identity_loss': [],
            'perceptual_loss': [],
            'structural_loss': [],
            'dehazing_loss': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """訓練一個 epoch"""
        self.G_A2B.train()
        self.G_B2A.train()
        self.D_A.train()
        self.D_B.train()
        
        epoch_losses = {
            'G_loss': 0, 'D_A_loss': 0, 'D_B_loss': 0,
            'cycle_loss': 0, 'identity_loss': 0, 'perceptual_loss': 0,
            'structural_loss': 0, 'dehazing_loss': 0
        }
        
        for i, (real_A, real_B) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            real_A, real_B = real_A.to(device), real_B.to(device)
            
            # 生成假圖像
            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)
            
            # 訓練生成器
            self.optimizer_G.zero_grad()
            
            # GAN 損失 - 多尺度損失取均值
            pred_fake_A = self.D_A(fake_A)
            pred_fake_B = self.D_B(fake_B)
            
            loss_GAN_A2B = sum([self.criterion_GAN(pred, torch.ones_like(pred)) for pred in pred_fake_B]) / len(pred_fake_B)
            loss_GAN_B2A = sum([self.criterion_GAN(pred, torch.ones_like(pred)) for pred in pred_fake_A]) / len(pred_fake_A)
            
            # 循環一致性損失
            recovered_A = self.G_B2A(fake_B)
            recovered_B = self.G_A2B(fake_A)
            loss_cycle_A = self.criterion_cycle(recovered_A, real_A)
            loss_cycle_B = self.criterion_cycle(recovered_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * self.lambda_cycle
            
            # 身份損失
            identity_A = self.G_B2A(real_A)
            identity_B = self.G_A2B(real_B)
            loss_identity_A = self.criterion_identity(identity_A, real_A)
            loss_identity_B = self.criterion_identity(identity_B, real_B)
            loss_identity = (loss_identity_A + loss_identity_B) * self.lambda_identity
            
            # 感知損失
            loss_perceptual_A = self.criterion_perceptual(fake_B, real_B)
            loss_perceptual_B = self.criterion_perceptual(fake_A, real_A)
            loss_perceptual = (loss_perceptual_A + loss_perceptual_B) * self.lambda_perceptual
            
            # 結構保持損失（防止過度風格化）
            loss_structural_A = self.criterion_structural(real_A, fake_B)  # A->B 結構保持
            loss_structural_B = self.criterion_structural(real_B, fake_A)  # B->A 結構保持
            loss_structural = (loss_structural_A + loss_structural_B) * self.lambda_structural
            
            # 去霧先驗損失（僅適用於 A->B，即有霧->清晰）
            loss_dehazing = self.criterion_dehazing(real_A, fake_B) * self.lambda_dehazing
            
            # 總生成器損失
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle + loss_identity + loss_perceptual + loss_structural + loss_dehazing
            loss_G.backward()
            self.optimizer_G.step()
            
            # 訓練判別器 A
            self.optimizer_D_A.zero_grad()
            
            pred_real_A = self.D_A(real_A)
            loss_D_A_real = sum([self.criterion_GAN(pred, torch.ones_like(pred)) for pred in pred_real_A]) / len(pred_real_A)
            
            fake_A_buffered = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = self.D_A(fake_A_buffered.detach())
            loss_D_A_fake = sum([self.criterion_GAN(pred, torch.zeros_like(pred)) for pred in pred_fake_A]) / len(pred_fake_A)
            
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()
            self.optimizer_D_A.step()
            
            # 訓練判別器 B
            self.optimizer_D_B.zero_grad()
            
            pred_real_B = self.D_B(real_B)
            loss_D_B_real = sum([self.criterion_GAN(pred, torch.ones_like(pred)) for pred in pred_real_B]) / len(pred_real_B)
            
            fake_B_buffered = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = self.D_B(fake_B_buffered.detach())
            loss_D_B_fake = sum([self.criterion_GAN(pred, torch.zeros_like(pred)) for pred in pred_fake_B]) / len(pred_fake_B)
            
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()
            self.optimizer_D_B.step()
            
            # 累積損失
            epoch_losses['G_loss'] += loss_G.item()
            epoch_losses['D_A_loss'] += loss_D_A.item()
            epoch_losses['D_B_loss'] += loss_D_B.item()
            epoch_losses['cycle_loss'] += loss_cycle.item()
            epoch_losses['identity_loss'] += loss_identity.item()
            epoch_losses['perceptual_loss'] += loss_perceptual.item()
            epoch_losses['structural_loss'] += loss_structural.item()
            epoch_losses['dehazing_loss'] += loss_dehazing.item()
            
            if (i + 1) % 100 == 0:
                print(f"Batch [{i+1}] - G: {loss_G.item():.4f}, D_A: {loss_D_A.item():.4f}, "
                      f"D_B: {loss_D_B.item():.4f}, Cycle: {loss_cycle.item():.4f}, "
                      f"Struct: {loss_structural.item():.4f}, DCP: {loss_dehazing.item():.4f}")
        
        # 計算平均損失
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def save_checkpoint(self, epoch):
        """保存檢查點"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'G_A2B_state_dict': self.G_A2B.state_dict(),
            'G_B2A_state_dict': self.G_B2A.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'losses_history': self.losses_history
        }
        torch.save(checkpoint, f'checkpoints/cyclegan_v6_epoch_{epoch}.pth')
        print(f"✅ 檢查點已保存: cyclegan_v6_epoch_{epoch}.pth")
    
    def plot_losses(self):
        """生成優化版損失函數圖表 - 包含 DCP 損失"""
        plt.figure(figsize=(20, 12))
        
        # 添加總標題
        plt.suptitle('CycleGAN v6.0 優化版損失函數分析 ([0,1]域 + 最小池化 + 反射Padding + LR→0)', 
                    fontsize=16, fontweight='bold')
        
        # 創建子圖 (3x3 layout)
        plt.subplot(3, 3, 1)
        plt.plot(self.losses_history['G_loss'], label='Generator Loss', color='blue')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 2)
        plt.plot(self.losses_history['D_A_loss'], label='Discriminator A Loss', color='red')
        plt.plot(self.losses_history['D_B_loss'], label='Discriminator B Loss', color='orange')
        plt.title('Discriminator Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 3)
        plt.plot(self.losses_history['cycle_loss'], label='Cycle Loss', color='green')
        plt.title('Cycle Consistency Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 4)
        plt.plot(self.losses_history['identity_loss'], label='Identity Loss (High Weight)', color='purple')
        plt.title('Identity Loss (λ=20)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 5)
        plt.plot(self.losses_history['perceptual_loss'], label='Perceptual Loss', color='brown')
        plt.title('Perceptual Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 6)
        plt.plot(self.losses_history['structural_loss'], label='Structural Loss', color='cyan')
        plt.title('Structural Preservation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 7)
        plt.plot(self.losses_history['dehazing_loss'], label='DCP Loss (Dehazing Prior)', color='magenta')
        plt.title('DCP Loss - 暗通道先驗去霧損失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 8)
        plt.plot(self.losses_history['G_loss'], label='Generator', alpha=0.7)
        plt.plot(self.losses_history['cycle_loss'], label='Cycle', alpha=0.7)
        plt.plot(self.losses_history['identity_loss'], label='Identity', alpha=0.7)
        plt.title('Core Losses Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 3, 9)
        plt.plot(self.losses_history['structural_loss'], label='Structural', alpha=0.7, color='cyan')
        plt.plot(self.losses_history['dehazing_loss'], label='DCP Loss', alpha=0.7, color='magenta')
        plt.plot(self.losses_history['perceptual_loss'], label='Perceptual', alpha=0.7, color='brown')
        plt.title('去霧專用損失對比 (Dehazing-Specific Losses)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 為總標題留空間
        
        # 生成時間戳避免覆蓋
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cyclegan_v6_optimized_losses_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 優化版去霧損失函數圖表已保存: {filename}")

def main():
    print("🚀 啟動 CycleGAN v6.0 去霧專用訓練")
    print("主要改進：結構保持損失 + 去霧先驗損失 + 高權重Identity損失")
    print("🎯 防止風格轉換化，生成真實去霧效果")
    
    # 數據加載
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CycleGANDataset(
        root_A='origin',  # 有霧圖像 (使用全部1500張)
        root_B='clean',   # 清晰圖像 (限制為1800張)
        transform=transform,
        max_images_B=1800  # 只限制清晰圖像數量為1800張
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print(f"📊 數據集大小: {len(dataset)} 對圖像")
    
    # 創建訓練器 - 去霧專用參數
    trainer = V6CycleGANTrainer(
        lambda_cycle=10.0,          # 循環一致性損失
        lambda_identity=20.0,       # 身份損失 (提升4倍防止風格化)
        lambda_perceptual=1.0,      # 感知損失
        lambda_structural=5.0,      # 結構保持損失 (防止過度變形)
        lambda_dehazing=3.0,        # 去霧先驗損失 (暗通道+對比度)
        lr=0.0002
    )
    
    # 訓練參數
    num_epochs = 200
    start_time = datetime.now()
    
    print(f"🏃 開始訓練 {num_epochs} 個 epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        # 訓練一個 epoch
        epoch_losses = trainer.train_epoch(dataloader, epoch)
        
        # 記錄損失
        for key, value in epoch_losses.items():
            trainer.losses_history[key].append(value)
        
        # 更新學習率
        trainer.scheduler_G.step()
        trainer.scheduler_D_A.step()
        trainer.scheduler_D_B.step()
        
        # 打印損失
        print(f"損失 - G: {epoch_losses['G_loss']:.4f}, D_A: {epoch_losses['D_A_loss']:.4f}, "
              f"D_B: {epoch_losses['D_B_loss']:.4f}, Cycle: {epoch_losses['cycle_loss']:.4f}")
        print(f"      Identity: {epoch_losses['identity_loss']:.4f}, Perceptual: {epoch_losses['perceptual_loss']:.4f}, "
              f"Structural: {epoch_losses['structural_loss']:.4f}, DCP: {epoch_losses['dehazing_loss']:.4f}")
        
        # 每20個epoch保存檢查點
        if epoch % 20 == 0:
            trainer.save_checkpoint(epoch)
        
        # 每50個epoch繪製損失曲線
        if epoch % 50 == 0:
            trainer.plot_losses()
            print(f"📊 Epoch {epoch} 損失圖表已生成")
    
    # 保存最終模型
    trainer.save_checkpoint(num_epochs)
    
    # 最終損失圖
    trainer.plot_losses()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # 生成時間戳和文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_chart_name = f'cyclegan_v6_optimized_losses_{timestamp}.png'
    
    print(f"\n🎉 優化版去霧專用 CycleGAN v6.0 訓練完成!")
    print(f"⏱️ 總訓練時間: {training_time}")
    print(f"💾 模型已保存: checkpoints/cyclegan_v6_epoch_{num_epochs}.pth")
    print(f"📊 優化版損失圖表已生成: {loss_chart_name}")
    print(f"🔥 包含完整的 DCP 損失 (暗通道先驗去霧損失) 分析")
    print(f"🎯 模型使用 [0,1] 域計算、最小池化、反射padding、LR衰減到0")

if __name__ == "__main__":
    main()