#!/usr/bin/env python3
"""
CycleGAN v6.0 多模型圖像去霧 Web 應用
支持下拉選單選擇不同的 v6 模型
"""

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import base64
import cv2
from datetime import datetime
import uuid
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'web_results'

# 創建必要資料夾
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 設備設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 可用的 v6 模型配置
V6_MODELS = {
    'v6_epoch_40': {
        'name': 'CycleGAN v6.0 - 平衡版本 (Epoch 40)',
        'path': 'checkpoints/cyclegan_v6_epoch_40.pth',
        'description': '適合日常使用的均衡去霧效果，處理速度快，效果穩定。',
        'badge': 'balanced'
    },
    'v6_epoch_120': {
        'name': 'CycleGAN v6.0 - 高級版本 (Epoch 120)',
        'path': 'checkpoints/cyclegan_v6_epoch_120.pth',
        'description': '推薦使用！更強的去霧能力，細節保持更好，適合大部分場景。',
        'badge': 'premium'
    },
    'v6_epoch_200': {
        'name': 'CycleGAN v6.0 - 完整版本 (Epoch 200)',
        'path': 'checkpoints/cyclegan_v6_epoch_200.pth',
        'description': '最強去霧效果，最佳細節還原，適合專業用途和高品質需求。',
        'badge': 'ultimate'
    }
}

def spectral_norm(module, name='weight', power_iterations=1):
    """使用 PyTorch 內建的 spectral normalization"""
    try:
        return torch.nn.utils.spectral_norm(module, name=name, n_power_iterations=power_iterations)
    except:
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

# 全域變數存儲模型
loaded_models = {}

def load_model(model_key):
    """動態載入指定的模型"""
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    if model_key not in V6_MODELS:
        raise ValueError(f"未知的模型: {model_key}")
    
    model_info = V6_MODELS[model_key]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"🔄 載入模型: {model_info['name']}")
    
    # 創建模型
    model = V6Generator()
    
    # 載入權重
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'G_A2B_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['G_A2B_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # 緩存模型
        loaded_models[model_key] = model
        
        print(f"✅ 成功載入: {model_info['name']}")
        return model
        
    except Exception as e:
        print(f"❌ 載入失敗: {str(e)}")
        raise

class OverlapInference:
    """重疊推理處理大圖像"""
    def __init__(self, tile_size=256, overlap=32):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
    def process_large_image(self, model, image_tensor):
        """處理大圖像"""
        B, C, H, W = image_tensor.shape
        
        if H <= self.tile_size and W <= self.tile_size:
            return model(image_tensor)
        
        # 計算需要的 tiles
        h_tiles = math.ceil(H / self.stride)
        w_tiles = math.ceil(W / self.stride)
        
        # 輸出圖像
        output = torch.zeros_like(image_tensor)
        weight_map = torch.zeros((H, W), device=image_tensor.device)
        
        for i in range(h_tiles):
            for j in range(w_tiles):
                # 計算tile位置
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = min(h_start + self.tile_size, H)
                w_end = min(w_start + self.tile_size, W)
                
                # 調整tile大小確保不超出邊界
                actual_h = h_end - h_start
                actual_w = w_end - w_start
                
                # 提取tile
                tile = image_tensor[:, :, h_start:h_end, w_start:w_end]
                
                # 如果tile太小，需要padding
                if actual_h < self.tile_size or actual_w < self.tile_size:
                    padded_tile = torch.zeros((B, C, self.tile_size, self.tile_size), 
                                            device=image_tensor.device)
                    padded_tile[:, :, :actual_h, :actual_w] = tile
                    tile_output = model(padded_tile)
                    tile_output = tile_output[:, :, :actual_h, :actual_w]
                else:
                    tile_output = model(tile)
                
                # 創建權重mask（中心權重更高）
                tile_weight = torch.ones((actual_h, actual_w), device=image_tensor.device)
                
                # 邊緣降權
                if i > 0:  # 不是第一行
                    fade_h = min(self.overlap, actual_h)
                    for k in range(fade_h):
                        weight = k / fade_h
                        tile_weight[k, :] *= weight
                        
                if j > 0:  # 不是第一列
                    fade_w = min(self.overlap, actual_w)
                    for k in range(fade_w):
                        weight = k / fade_w
                        tile_weight[:, k] *= weight
                
                # 累加到輸出
                output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight[None, None, :, :]
                weight_map[h_start:h_end, w_start:w_end] += tile_weight
        
        # 歸一化
        weight_map = torch.clamp(weight_map, min=1e-8)
        output = output / weight_map[None, None, :, :]
        
        return output

def preprocess_image(image_pil):
    """預處理圖像"""
    # 轉換為RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # 轉換為tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    return image_tensor

def postprocess_image(image_tensor):
    """後處理圖像"""
    # 反歸一化
    image_tensor = (image_tensor + 1) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # 轉換為PIL
    image_np = image_tensor.squeeze(0).cpu().detach().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * 255).astype(np.uint8)
    
    return Image.fromarray(image_np)

def process_image_with_model(image_pil, model_key):
    """使用指定模型處理圖像"""
    model = load_model(model_key)
    
    # 預處理
    input_tensor = preprocess_image(image_pil)
    
    # 推理
    with torch.no_grad():
        if input_tensor.shape[2] > 512 or input_tensor.shape[3] > 512:
            # 大圖像使用重疊推理
            overlap_inference = OverlapInference(tile_size=256, overlap=32)
            output_tensor = overlap_inference.process_large_image(model, input_tensor)
        else:
            # 小圖像直接處理
            output_tensor = model(input_tensor)
    
    # 後處理
    result_image = postprocess_image(output_tensor)
    return result_image

@app.route('/')
def index():
    """主頁"""
    return render_template('v6_optimized_index.html', models=V6_MODELS)

@app.route('/process', methods=['POST'])
def process():
    """處理圖像請求"""
    try:
        # 檢查文件
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '未選擇圖像文件'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名為空'})
        
        # 檢查模型選擇
        model_key = request.form.get('model', 'v6_epoch_120')
        if model_key not in V6_MODELS:
            return jsonify({'success': False, 'error': '無效的模型選擇'})
        
        # 讀取圖像
        image_pil = Image.open(io.BytesIO(file.read()))
        original_size = image_pil.size
        
        # 記錄處理時間
        start_time = datetime.now()
        
        # 處理圖像
        result_image = process_image_with_model(image_pil, model_key)
        
        # 調整輸出尺寸為原始尺寸
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
        
        end_time = datetime.now()
        processing_time = f"{(end_time - start_time).total_seconds():.2f}秒"
        
        # 轉換為base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        result_data_url = f"data:image/jpeg;base64,{img_str}"
        
        return jsonify({
            'success': True,
            'result_image': result_data_url,
            'model_name': V6_MODELS[model_key]['name'],
            'processing_time': processing_time,
            'image_size': f"{original_size[0]}×{original_size[1]}"
        })
        
    except Exception as e:
        print(f"處理錯誤: {str(e)}")
        return jsonify({'success': False, 'error': f'處理失敗: {str(e)}'})

if __name__ == '__main__':
    print(f"🚀 使用設備: {device}")
    print("🌟 啟動 CycleGAN V6.0 多模型去霧服務")
    print("📱 訪問: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)