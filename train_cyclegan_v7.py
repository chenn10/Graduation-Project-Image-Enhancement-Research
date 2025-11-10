#!/usr/bin/env python3
"""
CycleGAN v7.0 Enhanced - 全面改進版本
主要改進：
1. 修正結構損失配對：struct(real_A, fake_B) / struct(real_B, fake_A)
2. 使用 MultiScaleCycleGANDataset
3. DCP/Contrast 權重下調 + 邊界 gate
4. 調整超參數：lr_D = 8e-5, flip_prob = 0.02
5. 添加 TV loss (0.1 係數起步)
6. SelfAttention A/B 測試選項
7. 關閉前端 colormap 疊色
"""

import os
# Reduce OpenMP/MKL thread usage to avoid multiple runtime initialization on Windows
# This should be set before importing torch so linked libraries respect it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Allow duplicate OpenMP runtime to prevent crashes (temporary workaround for Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import math
from cyclegan_dataset import CycleGANDataset

class MultiScaleCycleGANDataset(CycleGANDataset):
    """多尺度 CycleGAN 數據集 - 隨機切取不同尺寸的 tile"""
    def __init__(self, root_A, root_B, transform=None, max_images=None, 
                 tile_sizes=[256, 384, 512], tile_prob=[0.6, 0.3, 0.1]):
        # 創建一個基本 transform 給父類，但實際不使用
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        super().__init__(root_A, root_B, base_transform, max_images)
        self.tile_sizes = tile_sizes
        self.tile_prob = tile_prob
        
        # 為不同尺寸創建 transform（調低 flip_prob）
        self.transforms = {}
        for size in tile_sizes:
            self.transforms[size] = transforms.Compose([
                transforms.Resize(int(size * 1.12)),  # 放大 12% 再裁剪
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.02),  # 降低 flip 機率
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __getitem__(self, index):
        # 隨機選擇 tile 尺寸
        tile_size = np.random.choice(self.tile_sizes, p=self.tile_prob)
        transform = self.transforms[tile_size]
        
        # 載入圖像（使用正確的屬性名）
        img_A = Image.open(self.images_A[index % len(self.images_A)]).convert('RGB')
        img_B = Image.open(self.images_B[index % len(self.images_B)]).convert('RGB')
        
        # 應用對應尺寸的 transform
        if transform:
            img_A = transform(img_A)
            img_B = transform(img_B)
        
        return img_A, img_B

# 設備設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# Ensure PyTorch uses a single thread for its CPU ops (helps avoid OpenMP runtime issues)
try:
    torch.set_num_threads(1)
except Exception:
    pass

# 創建必要資料夾
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('training_images', exist_ok=True)

def spectral_norm(module, name='weight', power_iterations=1):
    """使用 PyTorch 內建的 spectral normalization"""
    try:
        return torch.nn.utils.spectral_norm(module, name=name, n_power_iterations=power_iterations)
    except:
        return module

class TVLoss(nn.Module):
    """Total Variation Loss - 加強抑制條紋和棋盤格"""
    def __init__(self, weight=0.2):  # 提高權重以更好抑制 artifacts
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 計算水平和垂直方向的變化
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        tv_loss = self.weight * (tv_h + tv_w) / (batch_size * channels * height * width)
        return tv_loss

class HazeEstimator(nn.Module):
    """霧濃度估計器 - 評估圖像中霧的濃度"""
    def __init__(self):
        super(HazeEstimator, self).__init__()
        
    def estimate_haze_density(self, img):
        """
        估計霧濃度 (0-1)，基於多種指標：
        1. 暗通道先驗值
        2. 對比度
        3. 可見度
        4. 色彩飽和度
        """
        with torch.no_grad():  # 霧濃度計算不需要梯度
            # 確保在 [0,1] 範圍
            if img.min() < 0:
                img_01 = (img + 1.0) / 2.0
            else:
                img_01 = img
            
            # 計算暗通道
            dark_channel = torch.min(img_01, dim=1, keepdim=True)[0]
            # 使用 -max_pool2d(-x) 來模擬 min_pool2d
            dark_channel = -F.max_pool2d(-dark_channel, kernel_size=15, stride=1, padding=7)
            
            # 計算霧濃度指標，保持 batch 維度 [B]
            haze_density = torch.mean(dark_channel, dim=[1,2,3])
            
            return haze_density  # 返回 shape=[B]，不使用 squeeze()

class StructuralLoss(nn.Module):
    """修正版結構損失 - 確保正確的 domain 配對"""
    def __init__(self, window_size=11, channels=3):
        super(StructuralLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        
        # 創建高斯窗
        window = self.create_window(window_size, channels)
        self.register_buffer('window', window)
        
    def create_window(self, window_size, channels):
        # 創建 1D 高斯核
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        # 創建 2D 高斯窗
        _2D_window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0))
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        # 確保輸入在 [0,1] 範圍內
        if img1.min() < 0:
            img1 = (img1 + 1.0) / 2.0
        if img2.min() < 0:
            img2 = (img2 + 1.0) / 2.0
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def forward(self, real_A, fake_B, real_B, fake_A):
        """
        修正版結構損失配對：
        - struct(real_A, fake_B): A domain 的真實圖像與從 B 生成的假圖像
        - struct(real_B, fake_A): B domain 的真實圖像與從 A 生成的假圖像
        """
        # 正確的配對方式
        loss_A = 1.0 - self.ssim(real_A, fake_B)  # A→B 結構一致性
        loss_B = 1.0 - self.ssim(real_B, fake_A)  # B→A 結構一致性
        
        return (loss_A + loss_B) / 2.0

class SelfAttention(nn.Module):
    """使用 CBAM (Convolutional Block Attention Module) 替代原本的 Self-Attention。
    保持類名為 SelfAttention 以維持與現有程式碼相容，但內部實作為 CBAM。
    支援 with_attn 開關以關閉 attention。
    """
    def __init__(self, in_dim, activation=F.relu, with_attn=True, reduction=16, kernel_size=7):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn

        # Channel attention MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // reduction, in_dim, 1, bias=True)
        )

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size//2), bias=False),
            nn.Sigmoid()
        )

        # gamma for residual scaling (keeps similarity with original SelfAttention behavior)
        self.gamma = nn.Parameter(torch.zeros(1))

    def channel_att(self, x):
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        avg_out = self.mlp(avg)
        max_out = self.mlp(max_)
        out = torch.sigmoid(avg_out + max_out)
        return x * out

    def spatial_att(self, x):
        # channel-wise max and mean
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        mean_c = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([max_c, mean_c], dim=1)
        att = self.spatial(cat)
        return x * att

    def forward(self, x):
        if not self.with_attn:
            return x

        out = self.channel_att(x)
        out = self.spatial_att(out)

        # residual-style scaling similar to original SelfAttention
        return self.gamma * out + x

class ImprovedUpsample(nn.Module):
    """改進的上採樣模組 - 使用最近鄰避免棋盤格"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ImprovedUpsample, self).__init__()
        # 使用最近鄰上採樣 + 卷積替代 PixelShuffle
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    """殘差塊"""
    def __init__(self, channels, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(channels, channels, 3)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(channels, channels, 3)),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    """增強版生成器 - 支持 SelfAttention A/B 測試"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9, use_self_attention=True):
        super(Generator, self).__init__()
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(input_channels, 64, 7)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 殘差塊
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # 自注意力（可選）
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attention = SelfAttention(256, with_attn=True)
        else:
            self.self_attention = SelfAttention(256, with_attn=False)
        
        # 解碼器 - 使用改進的上採樣
        self.decoder = nn.Sequential(
            ImprovedUpsample(256, 128),
            nn.InstanceNorm2d(128),
            
            ImprovedUpsample(128, 64),
            nn.InstanceNorm2d(64),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        residual = self.residual_blocks(encoded)
        attended = self.self_attention(residual)
        output = self.decoder(attended)
        
        # 輸出亮度/對比度調整 - 防止偏暗
        # out = out * 1.1 + 0.05 (在 tanh 輸出範圍 [-1,1] 內調整)
        output = output * 1.1 + 0.05
        output = torch.clamp(output, -1.0, 1.0)  # 確保輸出範圍
        
        return output

class Discriminator(nn.Module):
    """判別器"""
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters):
            # 移除未使用的 normalize 參數，保持簡潔
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            # 移除 InstanceNorm 以減少假紋理
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class DCPLoss(nn.Module):
    """改進的 DCP 損失 - 支援真正的 per-image 自適應權重"""
    def __init__(self, patch_size=15, weight=0.3, w_min=0.1, w_max=0.5):
        super(DCPLoss, self).__init__()
        self.patch_size = patch_size
        self.weight = weight
        self.w_min = w_min  # 權重下界（保守設定）
        self.w_max = w_max  # 權重上界（保守設定）
        
    def forward(self, img, use_edge_gate=True, override_weight=None):
        # 確保在 [0,1] 範圍
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        
        # 計算暗通道
        dark_channel = torch.min(img, dim=1, keepdim=True)[0]
        # 使用 -max_pool2d(-x) 來模擬 min_pool2d
        dark_channel = -F.max_pool2d(-dark_channel, kernel_size=self.patch_size, 
                                    stride=1, padding=self.patch_size//2)
        
        # 邊界 gate - 減少邊緣區域的權重以避免 haloing (比例計算)
        if use_edge_gate:
            # 創建邊界遮罩 - 使用比例計算適配不同解析度
            b, c, h, w = img.shape
            mh, mw = max(16, h//8), max(16, w//8)  # 依圖像大小動態調整
            edge_gate = torch.ones_like(dark_channel)
            
            # 邊緣區域權重下調到 0.3
            edge_gate[:, :, :mh, :] *= 0.3
            edge_gate[:, :, -mh:, :] *= 0.3
            edge_gate[:, :, :, :mw] *= 0.3
            edge_gate[:, :, :, -mw:] *= 0.3
            
            dark_channel = dark_channel * edge_gate
        
        # 計算每張圖的 DCP 損失（保持 batch 維度）
        dcp_loss_per_image = torch.mean(dark_channel.view(dark_channel.size(0), -1), dim=1)  # shape=[B]
        
        # 使用自適應權重或默認權重
        if override_weight is not None and isinstance(override_weight, torch.Tensor):
            # per-image 權重：逐張乘權重後再平均
            weighted_loss = (override_weight * dcp_loss_per_image).mean()
        else:
            # 標量權重或默認權重
            weight = override_weight if override_weight is not None else self.weight
            weighted_loss = weight * dcp_loss_per_image.mean()
            
        return weighted_loss

class ContrastLoss(nn.Module):
    """改進的對比度損失 - 支援自適應權重與局部對比"""
    def __init__(self, weight=0.1, w_min=0.05, w_max=0.15):
        super(ContrastLoss, self).__init__()
        self.weight = weight
        self.w_min = w_min  # 權重下界（薄霧時保守）
        self.w_max = w_max  # 權重上界（濃霧時適度）
        
    def forward(self, img, override_weight=None):
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        
        # 計算局部對比度（避免全域過度增強）
        # 使用 3x3 kernel 計算局部標準差
        kernel_size = 3
        padding = kernel_size // 2
        
        # 計算局部均值
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size * kernel_size)
        local_mean = F.conv2d(img.mean(dim=1, keepdim=True), kernel, padding=padding)
        
        # 計算局部標準差
        img_gray = img.mean(dim=1, keepdim=True)
        local_var = F.conv2d((img_gray - local_mean) ** 2, kernel, padding=padding)
        local_std = torch.sqrt(local_var + 1e-6)
        
        # 計算每張圖的對比度損失（保持 batch 維度）
        contrast_loss_per_image = -torch.mean(local_std.view(local_std.size(0), -1), dim=1)  # shape=[B]
        
        # 使用自適應權重或默認權重
        if override_weight is not None and isinstance(override_weight, torch.Tensor):
            # per-image 權重：逐張乘權重後再平均
            weighted_loss = (override_weight * contrast_loss_per_image).mean()
        else:
            # 標量權重或默認權重
            weight = override_weight if override_weight is not None else self.weight
            weighted_loss = weight * contrast_loss_per_image.mean()
            
        return weighted_loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def sample_images(generator_AB, generator_BA, dataloader, device, epoch, save_path):
    """生成樣本圖像進行視覺檢查 - 動態適配 batch 大小"""
    generator_AB.eval()
    generator_BA.eval()
    
    with torch.no_grad():
        real_A, real_B = next(iter(dataloader))
        real_A, real_B = real_A.to(device), real_B.to(device)
        
        fake_B = generator_AB(real_A)
        fake_A = generator_BA(real_B)
        
        # 動態數量 - 避免 batch_size=1 崩潰
        n = min(4, real_A.size(0))  # 最多4張，適配實際 batch 大小
        
        # 創建對比圖像
        images = torch.cat([real_A[:n], fake_B[:n], real_B[:n], fake_A[:n]], dim=0)
        images = (images + 1.0) / 2.0
        
        # 動態網格布局
        total_images = 4 * n  # 4 類型 × n 張圖
        cols = n
        rows = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if n == 1:
            axes = axes.reshape(rows, 1)  # 確保是2D array
        
        for i in range(total_images):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            
            # 添加標題
            if i < n:
                ax.set_title(f'Real A {col+1}', fontsize=10)
            elif i < 2*n:
                ax.set_title(f'Fake B {col+1}', fontsize=10)
            elif i < 3*n:
                ax.set_title(f'Real B {col+1}', fontsize=10)
            else:
                ax.set_title(f'Fake A {col+1}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch:03d}_samples.png', 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

def train_cyclegan():
    """訓練 CycleGAN v7.0 Enhanced"""
    
    print("🚀 開始 CycleGAN v7.0 Enhanced 訓練")
    print("主要改進：")
    print("  • 修正結構損失配對")
    print("  • MultiScaleCycleGANDataset")
    print("  • 下調 DCP/Contrast 權重 + 邊界 gate")
    print("  • lr_D = 8e-5, flip_prob = 0.02")
    print("  • TV loss (0.1 係數)")
    print("  • SelfAttention A/B 測試選項")
    
    # 超參數 - 改進版本
    num_epochs = 200
    batch_size = 1  # 保持 batch_size=1 以確保穩定性與 per-image 權重
    lr_G = 2e-4     # 生成器學習率
    lr_D = 8e-5     # 判別器學習率（下調）
    lambda_cycle = 10.0
    lambda_identity = 5.0
    lambda_structural = 4.5
    lambda_dcp = 0.3        # DCP 基礎權重
    lambda_contrast = 0.1   # Contrast 權重進一步下調
    lambda_tv = 0.2         # TV loss 係數提高
    # 邊緣保留權重
    lambda_edge_preserve = 0.6
    
    # SelfAttention A/B 測試選項 (暫時關閉避免記憶體/條紋問題)
    use_self_attention = False  # 設為 False 來測試無 SelfAttention 版本
    
    # 數據加載 - 使用多尺度數據集，修正 A/B 語意
    print("📊 載入多尺度數據集...")
    print("   A=有霧(origin), B=清晰(clean)")
    dataset = MultiScaleCycleGANDataset(
        'origin', 'clean',  # 修正：A=有霧, B=清晰
        max_images=1500,  # 限制每個 domain 最多 1500 張圖像
        tile_sizes=[256, 384, 512],
        tile_prob=[0.6, 0.3, 0.1]
    )
    # Use num_workers=0 on Windows to avoid child-process DLL/runtime conflicts
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 初始化模型
    print("🏗️ 初始化模型...")
    generator_AB = Generator(use_self_attention=use_self_attention).to(device)
    generator_BA = Generator(use_self_attention=use_self_attention).to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)
    
    # 權重初始化
    generator_AB.apply(weights_init_normal)
    generator_BA.apply(weights_init_normal)
    discriminator_A.apply(weights_init_normal)
    discriminator_B.apply(weights_init_normal)
    
    # 損失函數
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    structural_loss = StructuralLoss().to(device)
    dcp_loss = DCPLoss(weight=lambda_dcp, w_min=0.1, w_max=0.20)  # 將 w_max 降至 0.20
    contrast_loss = ContrastLoss(weight=lambda_contrast, w_min=0.05, w_max=0.15)  # 對比度也支援自適應
    tv_loss = TVLoss(weight=lambda_tv)
    haze_estimator = HazeEstimator().to(device)
    
    # 優化器 - 調整學習率
    optimizer_G = optim.Adam(
        itertools.chain(generator_AB.parameters(), generator_BA.parameters()),
        lr=lr_G, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=lr_D, betas=(0.5, 0.999))
    
    # 學習率調度
    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    # -----------------
    # Resume from latest checkpoint if exists
    # -----------------
    import glob

    def find_latest_checkpoint():
        patterns = [
            'checkpoints/cyclegan_v7_enhanced_no_attn_epoch_*.pth',
            'checkpoints/cyclegan_v7_enhanced_epoch_*.pth'
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        if not files:
            return None
        # sort by epoch number parsed from filename
        def epoch_from_name(f):
            try:
                return int(os.path.basename(f).split('_epoch_')[1].split('.pth')[0])
            except Exception:
                return 0
        files.sort(key=epoch_from_name)
        return files[-1]

    start_epoch = 0
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"🔁 發現 checkpoint {latest_ckpt}，嘗試載入以繼續訓練（允許部分權重不匹配）")
        try:
            ck = torch.load(latest_ckpt, map_location=device)
        except Exception as e:
            print(f"❌ 載入 checkpoint 檔案失敗: {e}，將從頭開始訓練")
            latest_ckpt = None
            start_epoch = 0

    if latest_ckpt is not None:
        # Load model weights with strict=False so mismatched attention modules won't block loading.
        def try_load_model(model, key_name):
            if key_name not in ck:
                print(f"⚠️ checkpoint 中沒有 {key_name} 權重，跳過")
                return
            state = ck[key_name]
            try:
                res = model.load_state_dict(state, strict=False)
                missing = res.missing_keys if hasattr(res, 'missing_keys') else []
                unexpected = res.unexpected_keys if hasattr(res, 'unexpected_keys') else []
                if missing:
                    print(f"⚠️ 在載入 {key_name} 時有缺失的參數 ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
                if unexpected:
                    print(f"⚠️ 在載入 {key_name} 時有多餘的參數 ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
            except Exception as e:
                print(f"❌ 載入 {key_name} 權重失敗: {e}")

        try:
            try_load_model(generator_AB, 'generator_AB')
            try_load_model(generator_BA, 'generator_BA')
            try_load_model(discriminator_A, 'discriminator_A')
            try_load_model(discriminator_B, 'discriminator_B')

            # try to load optimizer states (may fail across PyTorch versions) — optional
            try:
                if 'optimizer_G' in ck:
                    try:
                        optimizer_G.load_state_dict(ck['optimizer_G'])
                    except Exception as e:
                        print(f"⚠️ 無法載入 optimizer_G state: {e}")
                if 'optimizer_D_A' in ck:
                    try:
                        optimizer_D_A.load_state_dict(ck['optimizer_D_A'])
                    except Exception as e:
                        print(f"⚠️ 無法載入 optimizer_D_A state: {e}")
                if 'optimizer_D_B' in ck:
                    try:
                        optimizer_D_B.load_state_dict(ck['optimizer_D_B'])
                    except Exception as e:
                        print(f"⚠️ 無法載入 optimizer_D_B state: {e}")
            except Exception:
                pass

            start_epoch = int(ck.get('epoch', 0))
            print(f"✅ 嘗試從 checkpoint 恢復，start_epoch 設為 {start_epoch}（部分參數若不匹配則被忽略）")
        except Exception as e:
            print(f"❌ 在處理 checkpoint 時發生錯誤: {e}，將從頭開始訓練")
            start_epoch = 0
    
    # 訓練循環
    print(f"🎯 開始訓練 {num_epochs} epochs...")
    print(f"   SelfAttention: {'啟用' if use_self_attention else '關閉'}")
    print(f"   自適應權重: DCP [{dcp_loss.w_min:.2f}, {dcp_loss.w_max:.2f}], Contrast [{contrast_loss.w_min:.2f}, {contrast_loss.w_max:.2f}]")
    
    # 暖啟動參數
    warmup_epochs = 5  # 前 5 個 epoch 做暖啟
    
    for epoch in range(num_epochs):
        generator_AB.train()
        generator_BA.train()
        discriminator_A.train()
        discriminator_B.train()
        
        total_loss_G = 0
        total_loss_D = 0
        
        # 暖啟動權重縮放（前幾個 epoch 減緩自適應權重）
        warmup_scale = min(1.0, (epoch + 1) / warmup_epochs)
        
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A, real_B = real_A.to(device), real_B.to(device)
            
            # 動態計算判別器輸出尺寸（多尺度數據集需要）
            with torch.no_grad():
                sample_output = discriminator_A(real_A)
                output_size = sample_output.shape[2:]  # 獲取當前 batch 的 H, W 尺寸
            
            # 創建真實和虛假標籤（使用當前batch的尺寸）
            valid = torch.ones((real_A.size(0), 1, *output_size), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, *output_size), requires_grad=False).to(device)
            
            # -----------------
            # 訓練生成器
            # -----------------
            
            optimizer_G.zero_grad()
            
            # 生成假圖像
            fake_B_raw = generator_AB(real_A)
            fake_A = generator_BA(real_B)
            
            # 強霧 Gamma 校正 (1.05~1.25) - 基於霧濃度動態調整
            with torch.no_grad():
                haze_density_for_gamma = haze_estimator.estimate_haze_density(real_A)
                # 使用門檻 0.7，將 Gamma 範圍設為 1.05 ~ 1.25
                gamma_threshold = 0.7
                gamma_min = 1.05
                gamma_max = 1.25
                # 線性映射 haze_density 從 [gamma_threshold, 1.0] 到 [gamma_min, gamma_max]
                scaled = torch.clamp((haze_density_for_gamma - gamma_threshold) / (1.0 - gamma_threshold), 0.0, 1.0)
                gamma_values = torch.where(
                    haze_density_for_gamma > gamma_threshold,
                    gamma_min + (gamma_max - gamma_min) * scaled,
                    torch.ones_like(haze_density_for_gamma)
                )
            
            # 應用 gamma 校正
            fake_B_normalized = (fake_B_raw + 1.0) / 2.0  # [-1,1] → [0,1]
            gamma_corrected = torch.pow(fake_B_normalized.clamp(1e-6, 1.0), 
                                      gamma_values.view(-1, 1, 1, 1))
            fake_B = gamma_corrected * 2.0 - 1.0  # [0,1] → [-1,1]
            
            # GAN 損失
            pred_fake_B = discriminator_B(fake_B)
            pred_fake_A = discriminator_A(fake_A)
            loss_GAN_AB = criterion_GAN(pred_fake_B, valid)
            loss_GAN_BA = criterion_GAN(pred_fake_A, valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            
            # 循環一致性損失
            reconstructed_A = generator_BA(fake_B)
            reconstructed_B = generator_AB(fake_A)
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)
            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # 身份損失
            identity_A = generator_BA(real_A)
            identity_B = generator_AB(real_B)
            loss_identity_A = criterion_identity(identity_A, real_A)
            loss_identity_B = criterion_identity(identity_B, real_B)
            loss_identity = (loss_identity_A + loss_identity_B) / 2
            
            # 修正版結構損失配對
            loss_structural = structural_loss(real_A, fake_B, real_B, fake_A)
            
            # 霧濃度分級 (離散化) - 三段式映射
            haze_density_A = haze_estimator.estimate_haze_density(real_A).clamp(0, 1)  # shape=[B]
            
            # 連續權重模式 - 使用冪函數平滑調整
            # DCP 權重: 0.15 + 0.3 * (haze_density ** 1.2)
            adaptive_dcp_weight = 0.15 + 0.3 * torch.pow(haze_density_A, 1.2)
            # 限制在 DCPLoss 的 [w_min, w_max] 範圍內以避免過激權重
            adaptive_dcp_weight = adaptive_dcp_weight.clamp(dcp_loss.w_min, dcp_loss.w_max)
            
            # Contrast 權重: 基於霧濃度的連續調整
            adaptive_contrast_weight = 0.05 + 0.08 * torch.pow(haze_density_A, 1.1)
            
            # 中霧專屬補償（紋理保留）- 在中等霧濃度時啟用
            edge_loss = 0.0
            mid_haze_mask = (haze_density_A >= 0.25) & (haze_density_A < 0.7)
            if mid_haze_mask.any():
                # 計算邊緣紋理保留損失
                edge_map_A = torch.abs(real_A[:, :, 1:, :] - real_A[:, :, :-1, :]).mean(dim=1, keepdim=True)
                edge_fake_B = torch.abs(fake_B[:, :, 1:, :] - fake_B[:, :, :-1, :]).mean(dim=1, keepdim=True)
                edge_loss = F.l1_loss(edge_fake_B[:, :, :edge_map_A.size(2), :], edge_map_A)
            
            # 應用暖啟動縮放
            adaptive_dcp_weight = adaptive_dcp_weight * warmup_scale
            adaptive_contrast_weight = adaptive_contrast_weight * warmup_scale
            
            # DCP 損失 - 只施加在 A→B (有霧→清晰) 方向，使用自適應權重
            loss_dcp_AB = dcp_loss(fake_B, use_edge_gate=True, override_weight=adaptive_dcp_weight)
            loss_dcp_total = loss_dcp_AB  # 不對 fake_A 施加 DCP
            
            # 對比度損失 - 只施加在 A→B (有霧→清晰) 方向，使用自適應權重
            loss_contrast_AB = contrast_loss(fake_B, override_weight=adaptive_contrast_weight)
            loss_contrast_total = loss_contrast_AB  # 不對 fake_A 施加對比度約束
            
            # TV 損失
            loss_tv_AB = tv_loss(fake_B)
            loss_tv_BA = tv_loss(fake_A)
            loss_tv_total = (loss_tv_AB + loss_tv_BA) / 2
            
            # 總生成器損失
            loss_G_total = (
                loss_GAN +
                lambda_cycle * loss_cycle +
                lambda_identity * loss_identity +
                lambda_structural * loss_structural +
                loss_dcp_total +
                loss_contrast_total +
                loss_tv_total +
                lambda_edge_preserve * edge_loss  # 中霧補償項 - 保留遠景紋理
            )
            
            loss_G_total.backward()
            optimizer_G.step()
            
            # -----------------
            # 訓練判別器 A
            # -----------------
            
            optimizer_D_A.zero_grad()
            
            # 真實圖像損失
            pred_real_A = discriminator_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, valid)
            
            # 假圖像損失
            pred_fake_A = discriminator_A(fake_A.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, fake)
            
            # 總判別器 A 損失
            loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
            
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # -----------------
            # 訓練判別器 B
            # -----------------
            
            optimizer_D_B.zero_grad()
            
            # 真實圖像損失
            pred_real_B = discriminator_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, valid)
            
            # 假圖像損失
            pred_fake_B = discriminator_B(fake_B.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake)
            
            # 總判別器 B 損失
            loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2
            
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # 累積損失
            total_loss_G += loss_G_total.item()
            total_loss_D += (loss_D_A.item() + loss_D_B.item())
            
            # 打印進度（含離散分級信息）
            if i % 10 == 0:
                current_haze = haze_density_A.mean().item() if len(haze_density_A.shape) > 0 else haze_density_A.item()
                current_dcp_w = adaptive_dcp_weight.mean().item() if len(adaptive_dcp_weight.shape) > 0 else adaptive_dcp_weight.item()
                
                # 統計連續權重範圍
                dcp_min = adaptive_dcp_weight.min().item() if len(adaptive_dcp_weight.shape) > 0 else adaptive_dcp_weight.item()
                dcp_max = adaptive_dcp_weight.max().item() if len(adaptive_dcp_weight.shape) > 0 else adaptive_dcp_weight.item()
                
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] "
                      f"Loss_G: {loss_G_total.item():.4f} "
                      f"Loss_D: {(loss_D_A.item() + loss_D_B.item()):.4f} "
                      f"Cycle: {loss_cycle.item():.4f} "
                      f"DCP: {loss_dcp_total.item():.4f}(w={current_dcp_w:.3f}±{dcp_max-dcp_min:.2f}) "
                      f"Edge: {edge_loss:.4f} "
                      f"Haze: {current_haze:.3f} (連續模式)")
        
        # 更新學習率
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        # 每20個epoch保存模型和樣本
        if (epoch + 1) % 20 == 0:
            checkpoint_name = f"cyclegan_v7_enhanced_epoch_{epoch+1}.pth"
            if not use_self_attention:
                checkpoint_name = f"cyclegan_v7_enhanced_no_attn_epoch_{epoch+1}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'generator_AB': generator_AB.state_dict(),
                'generator_BA': generator_BA.state_dict(),
                'discriminator_A': discriminator_A.state_dict(),
                'discriminator_B': discriminator_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
                'use_self_attention': use_self_attention
            }, f'checkpoints/{checkpoint_name}')
            
            print(f"✅ 已保存模型: {checkpoint_name}")
            
            # 生成樣本圖像
            sample_images(generator_AB, generator_BA, dataloader, device, 
                         epoch + 1, 'training_images')
        
        # 每個epoch的統計
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D = total_loss_D / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成 - "
              f"平均 Loss_G: {avg_loss_G:.4f}, 平均 Loss_D: {avg_loss_D:.4f}")
    
    print("🎉 訓練完成！")

if __name__ == "__main__":
    train_cyclegan()
