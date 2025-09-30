#!/usr/bin/env python3
"""
CycleGAN 圖像去霧 Web 應用
使用 cyclegan_epoch_45_backup.pth 模型的單模型去霧服務
"""

import os
import io
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import base64
import cv2
from datetime import datetime
import uuid

from generator import ResNetGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'web_results'

# 創建必要資料夾
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 全域變數存儲模型
model = None
device = None

def load_model():
    """載入CycleGAN模型"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建模型
    model = ResNetGenerator()
    
    # 載入權重
    checkpoint_path = 'checkpoints/cyclegan_epoch_45_backup.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 根據檔案格式載入state_dict
    if 'G_A2B_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['G_A2B_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("✅ 模型載入成功！")

def pad_to_square(image_pil, target_size=256):
    """將圖像填充成正方形，保持縱橫比"""
    width, height = image_pil.size
    max_side = max(width, height)
    
    # 如果已經是正方形且尺寸正確，直接返回
    if width == height == target_size:
        return image_pil
    
    # 創建正方形背景（白色）
    new_image = Image.new('RGB', (max_side, max_side), (255, 255, 255))
    
    # 計算貼上位置（居中）
    paste_x = (max_side - width) // 2
    paste_y = (max_side - height) // 2
    
    # 貼上原圖
    new_image.paste(image_pil, (paste_x, paste_y))
    
    # resize到目標尺寸
    return new_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

def unpad_from_square(output_image, original_size, target_size=256):
    """從正方形輸出中提取原始比例的圖像"""
    original_width, original_height = original_size
    
    # 如果原圖已是正方形，直接resize
    if original_width == original_height:
        return output_image.resize(original_size, Image.Resampling.LANCZOS)
    
    # 計算原圖在padding後的實際尺寸
    max_side = max(original_width, original_height)
    scale = target_size / max_side
    
    # 計算原圖在256x256中的實際位置和尺寸
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)
    
    # 計算裁切位置（居中）
    crop_x = (target_size - scaled_width) // 2
    crop_y = (target_size - scaled_height) // 2
    
    # 裁切出原圖部分
    cropped = output_image.crop((crop_x, crop_y, crop_x + scaled_width, crop_y + scaled_height))
    
    # resize回原始尺寸
    return cropped.resize(original_size, Image.Resampling.LANCZOS)

def process_image(image_pil):
    """使用模型處理圖像"""
    original_size = image_pil.size
    
    # 使用padding預處理保持縱橫比
    padded_image = pad_to_square(image_pil, 256)
    
    # 轉換為tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(padded_image).unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
        # 轉換回圖像
        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = output_tensor * 0.5 + 0.5  # 反正規化
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # 轉為PIL圖像
        output_image = transforms.ToPILImage()(output_tensor)
        
        # 去除padding並恢復原始縱橫比
        output_image = unpad_from_square(output_image, original_size, 256)
        
        return output_image

@app.route('/')
def index():
    """主頁面"""
    return render_template('single_model_index.html')

@app.route('/process', methods=['POST'])
def process_uploaded_image():
    """處理上傳的圖像"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '沒有上傳圖像'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '沒有選擇文件'}), 400
        
        # 檢查檔案類型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': '不支援的檔案格式'}), 400
        
        # 載入圖像
        image_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        original_filename = secure_filename(file.filename)
        
        # 處理圖像
        result_image = process_image(image_pil)
        
        # 保存結果
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_image.save(result_path, format='PNG')
        
        # 轉換為base64用於前端顯示
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        result_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'original_filename': original_filename,
            'result_filename': result_filename,
            'image_size': f"{image_pil.size[0]}×{image_pil.size[1]}",
            'result_image': f"data:image/png;base64,{result_base64}"
        })
        
    except Exception as e:
        print(f"處理錯誤: {e}")
        return jsonify({'error': f'處理失敗: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """下載處理結果"""
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下載失敗: {str(e)}'}), 500

if __name__ == '__main__':
    print("正在載入CycleGAN去霧模型...")
    load_model()
    print("🚀 模型載入完成，啟動Web服務...")
    app.run(host='0.0.0.0', port=5000, debug=True)