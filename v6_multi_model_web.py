#!/usr/bin/env python3
"""
CycleGAN v7.0 Enhanced Web 應用
使用三級離散霧度分級的 v7 增強版模型
"""

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template_string
import base64
import glob

# 解決 spectral norm 問題
def spectral_norm(module, name='weight', power_iterations=1):
    try:
        return torch.nn.utils.spectral_norm(module, name=name, n_power_iterations=power_iterations)
    except:
        return module

class SelfAttention(nn.Module):
    """自注意力機制 - v7 版本"""
    def __init__(self, in_dim, activation=F.relu, with_attn=False):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        
        if self.with_attn:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        if not self.with_attn:
            return x
            
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        out = self.gamma * out + x
        return out

class ImprovedUpsample(nn.Module):
    """改進的上採樣模組 - v7 版本"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ImprovedUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    """殘差塊 - v7 版本"""
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

class V7Generator(nn.Module):
    """CycleGAN v7 Enhanced 生成器"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9, use_self_attention=False):
        super(V7Generator, self).__init__()
        
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
        return output

# 全域變數
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = None
current_model_path = None

def get_all_v7_models():
    """獲取所有可用的 v7 模型"""
    v7_models = glob.glob('checkpoints/cyclegan_v7_enhanced_no_attn_epoch_*.pth')
    if not v7_models:
        v7_models = glob.glob('checkpoints/cyclegan_v7_enhanced_epoch_*.pth')
    
    if not v7_models:
        return []
    
    # 按 epoch 數字排序
    v7_models.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    return v7_models

def find_latest_v7_model():
    """尋找最新的 v7 模型"""
    v7_models = get_all_v7_models()
    
    if not v7_models:
        print("❌ 找不到任何 v7 模型檔案")
        return None
    
    latest_model = v7_models[-1]
    print(f"🔍 找到 v7 模型: {latest_model}")
    return latest_model

def load_v7_model(model_path=None):
    """載入 v7 模型"""
    global generator, current_model_path
    
    try:
        print("🔄 載入 CycleGAN v7 Enhanced 模型...")
        
        # 如果沒指定模型，使用最新的
        if model_path is None:
            model_path = find_latest_v7_model()
        
        if model_path is None:
            return False, "找不到任何 v7 模型檔案"
        
        # 初始化生成器
        generator = V7Generator(use_self_attention=False).to(device)
        
        # 載入權重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 載入 generator_AB (有霧→清晰)
        if 'generator_AB' in checkpoint:
            generator.load_state_dict(checkpoint['generator_AB'])
            current_model_path = model_path
            print(f"✅ 成功載入 v7 模型: {model_path}")
        else:
            print("❌ 模型檔案中沒有 generator_AB")
            return False, "模型檔案中沒有 generator_AB"
        
        generator.eval()
        return True, model_path
        
    except Exception as e:
        print(f"❌ v7 模型載入失敗: {e}")
        return False, str(e)

def process_image_v7(image_pil):
    """使用 v7 模型處理圖像"""
    global generator
    
    try:
        # 保持比例的圖像轉換
        original_size = image_pil.size
        
        # 計算適當的處理尺寸（適配模型）
        def get_processing_size(w, h, target_size=512):
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 確保能被32整除
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            
            # 最小尺寸保證
            new_w = max(new_w, 256)
            new_h = max(new_h, 256)
            
            return new_w, new_h
        
        proc_w, proc_h = get_processing_size(original_size[0], original_size[1])
        
        # 圖像轉換
        transform = transforms.Compose([
            transforms.Resize((proc_h, proc_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 處理圖像
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # 轉換回 PIL
        output_numpy = output_tensor.squeeze().cpu().numpy()
        output_numpy = (output_numpy + 1.0) / 2.0
        output_numpy = np.transpose(output_numpy, (1, 2, 0))
        output_numpy = np.clip(output_numpy * 255, 0, 255).astype(np.uint8)
        
        result_image = Image.fromarray(output_numpy)
        
        # 調整回原始尺寸
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
        
        return result_image
        
    except Exception as e:
        print(f"❌ v7 圖像處理失敗: {e}")
        return None

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CycleGAN v7.0 Enhanced 去霧系統</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #4a5568; margin: 0; font-size: 2.5em; }
        .header p { color: #718096; font-size: 1.1em; margin: 10px 0; }
        
        .upload-section { 
            background: #f7fafc; 
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            border: 2px dashed #cbd5e0;
            transition: all 0.3s ease;
        }
        .upload-section:hover { border-color: #4299e1; background: #ebf8ff; }
        
        .upload-area { 
            text-align: center; 
            cursor: pointer; 
            padding: 40px;
            border-radius: 8px;
        }
        
        .btn { 
            background: linear-gradient(135deg, #4299e1, #3182ce); 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px; 
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
        }
        .btn:hover { 
            background: linear-gradient(135deg, #3182ce, #2c5282); 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(66, 153, 225, 0.4);
        }
        .btn:disabled { 
            background: #a0aec0; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        
        .results { 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
            display: none; 
            margin-top: 20px;
        }
        
        .image-comparison { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-top: 20px; 
        }
        
        .image-container { 
            text-align: center; 
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .image-container h4 { 
            margin-top: 0; 
            color: #2d3748;
            font-size: 1.2em;
        }
        .image-container img { 
            max-width: 100%; 
            border-radius: 8px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .image-container img:hover { transform: scale(1.05); }
        
        .loading { 
            text-align: center; 
            padding: 40px; 
            display: none; 
            background: white;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .spinner { 
            border: 4px solid #f3f3f3; 
            border-top: 4px solid #4299e1; 
            border-radius: 50%; 
            width: 50px; 
            height: 50px; 
            animation: spin 1s linear infinite; 
            margin: 0 auto 20px; 
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .status { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            text-align: center; 
            font-weight: bold;
        }
        .status.success { 
            background: #c6f6d5; 
            color: #22543d; 
            border: 1px solid #9ae6b4; 
        }
        .status.error { 
            background: #fed7d7; 
            color: #742a2a; 
            border: 1px solid #fc8181; 
        }
        
        .model-info {
            background: #edf2f7;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .model-info h3 {
            color: #2d3748;
            margin: 0 0 10px 0;
        }
        
        .feature-list {
            color: #4a5568;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .model-selector {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .model-selector select {
            padding: 10px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            background: white;
            font-size: 14px;
            color: #2d3748;
            min-width: 200px;
            margin-right: 10px;
        }
        
        .model-selector select:focus {
            outline: none;
            border-color: #4299e1;
        }
        
        .btn-switch {
            background: linear-gradient(135deg, #38b2ac, #319795);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
            transition: all 0.3s ease;
        }
        
        .btn-switch:hover {
            background: linear-gradient(135deg, #319795, #2c7a7b);
            transform: translateY(-1px);
        }
        
        @media (max-width: 768px) { 
            .image-comparison { grid-template-columns: 1fr; }
            .container { padding: 20px; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 CycleGAN v7.0 Enhanced</h1>
            <p>三級離散霧度分級 • 智能去霧系統</p>
        </div>
        
        <div class="model-selector">
            <h3>🔧 模型選擇</h3>
            <select id="modelSelect">
                <option value="">載入可用模型...</option>
            </select>
            <button class="btn-switch" onclick="switchModel()">切換模型</button>
            <p id="currentModel" style="margin-top: 10px; font-size: 0.9em; color: #666;">
                當前模型: 載入中...
            </p>
        </div>
        
        <div class="model-info">
            <h3>🚀 v7 增強版特色</h3>
            <div class="feature-list">
                ✨ 連續權重模式 (平滑霧度調整)<br>
                � 強霧 Gamma 校正 (1.1~1.3)<br>
                💡 生成器亮度優化 (防偏暗)<br>
                🛡️ 邊緣紋理補償機制<br>
                🔧 改進的上採樣避免棋盤效應<br>
                📊 多尺度訓練數據集
            </div>
        </div>
        
        <div class="upload-section">
            <h3>📁 上傳有霧圖像</h3>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p style="font-size: 1.2em; margin: 0;">📷 點擊此處選擇圖像文件</p>
                <p style="color: #666; font-size: 14px; margin: 10px 0 0 0;">支持 JPG, PNG 等格式，最大 16MB</p>
            </div>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <br><br>
            <button class="btn" id="processBtn" onclick="processImage()" disabled>🚀 開始 v7 去霧處理</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p id="loadingText">正在使用 v7 Enhanced 模型處理圖像...</p>
        </div>
        
        <div class="results" id="results">
            <h3>📊 v7 Enhanced 去霧結果</h3>
            <div class="image-comparison">
                <div class="image-container">
                    <h4>🌫️ 原始有霧圖像</h4>
                    <img id="originalImg" alt="原始圖像">
                </div>
                <div class="image-container">
                    <h4>✨ v7 去霧結果</h4>
                    <img id="resultImg" alt="v7 去霧結果">
                </div>
            </div>
        </div>
        
        <div id="status" class="status" style="display: none;"></div>
    </div>

    <script>
        let selectedFile = null;
        
        // 頁面載入時獲取可用模型
        document.addEventListener('DOMContentLoaded', function() {
            loadAvailableModels();
        });
        
        function loadAvailableModels() {
            fetch('/get_models')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const select = document.getElementById('modelSelect');
                    select.innerHTML = '';
                    
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.path;
                        option.textContent = model.name;
                        if (model.is_current) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                    
                    // 更新當前模型顯示
                    if (data.current_model) {
                        const currentEpoch = data.current_model.split('_epoch_')[1].split('.')[0];
                        document.getElementById('currentModel').textContent = 
                            `當前模型: v7 Enhanced Epoch ${currentEpoch}`;
                    }
                } else {
                    showStatus('載入模型列表失敗', 'error');
                }
            })
            .catch(error => {
                showStatus('載入模型列表時發生錯誤', 'error');
                console.error('Error:', error);
            });
        }
        
        function switchModel() {
            const select = document.getElementById('modelSelect');
            const selectedModel = select.value;
            
            if (!selectedModel) {
                showStatus('請選擇一個模型', 'error');
                return;
            }
            
            showStatus('正在切換模型...', 'success');
            
            fetch('/switch_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_path: selectedModel
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const currentEpoch = data.current_model.split('_epoch_')[1].split('.')[0];
                    document.getElementById('currentModel').textContent = 
                        `當前模型: v7 Enhanced Epoch ${currentEpoch}`;
                    showStatus('模型切換成功！', 'success');
                } else {
                    showStatus(`模型切換失敗: ${data.error}`, 'error');
                }
            })
            .catch(error => {
                showStatus('切換模型時發生錯誤', 'error');
                console.error('Error:', error);
            });
        }
        
        document.getElementById('fileInput').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('processBtn').disabled = false;
                document.getElementById('results').style.display = 'none';
                showStatus(`已選擇文件: ${file.name}`, 'success');
            }
        });
        
        function processImage() {
            if (!selectedFile) {
                showStatus('請先選擇圖像文件', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;
            
            fetch('/process_v7', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
                
                if (data.success) {
                    document.getElementById('originalImg').src = data.original_image;
                    document.getElementById('resultImg').src = data.result_image;
                    document.getElementById('results').style.display = 'block';
                    showStatus('v7 Enhanced 去霧處理完成！', 'success');
                } else {
                    showStatus(`v7 處理失敗: ${data.error}`, 'error');
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
                showStatus('v7 處理時發生錯誤', 'error');
                console.error('Error:', error);
            });
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.className = `status ${type}`;
            status.textContent = message;
            status.style.display = 'block';
            setTimeout(() => {
                status.style.display = 'none';
            }, 4000);
        }
    </script>
</body>
</html>
    ''')

@app.route('/get_models', methods=['GET'])
def get_available_models():
    """獲取所有可用的 v7 模型"""
    try:
        v7_models = get_all_v7_models()
        models_info = []
        
        for model_path in v7_models:
            # 提取 epoch 數字
            epoch = int(model_path.split('_epoch_')[1].split('.')[0])
            model_name = f"v7 Enhanced Epoch {epoch}"
            
            models_info.append({
                'path': model_path,
                'name': model_name,
                'epoch': epoch,
                'is_current': model_path == current_model_path
            })
        
        return jsonify({
            'success': True,
            'models': models_info,
            'current_model': current_model_path
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """切換 v7 模型"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'success': False, 'error': '請指定模型路徑'})
        
        success, message = load_v7_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'成功切換到模型: {model_path}',
                'current_model': current_model_path
            })
        else:
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_v7', methods=['POST'])
def process_v7_image():
    """使用 v7 模型處理上傳的圖像"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '沒有上傳圖像'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '沒有選擇文件'}), 400
        
        print(f"🎯 使用 v7 Enhanced 處理圖像: {file.filename}")
        
        # 載入圖像
        image_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 使用 v7 模型處理
        result_image = process_image_v7(image_pil)
        
        if result_image is None:
            return jsonify({'success': False, 'error': 'v7 圖像處理失敗'}), 500
        
        # 轉為base64
        def pil_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        return jsonify({
            'success': True,
            'original_image': pil_to_base64(image_pil),
            'result_image': pil_to_base64(result_image),
            'model_version': 'v7.0 Enhanced'
        })
        
    except Exception as e:
        print(f"❌ v7 處理錯誤: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 載入 CycleGAN v7.0 Enhanced 去霧系統...")
    success, message = load_v7_model()
    if success:
        print(f"🚀 啟動 v7 Enhanced 去霧服務 - {message}")
        app.run(host='0.0.0.0', port=5007, debug=True)
    else:
        print(f"❌ v7 模型載入失敗，無法啟動服務: {message}")
