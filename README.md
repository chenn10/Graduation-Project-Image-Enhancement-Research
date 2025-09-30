# CycleGAN 去霧系統

這是一個基於CycleGAN的圖像去霧系統，能夠將有霧圖像轉換為清晰圖像。

## 系統特點

- **CycleGAN架構**: 使用無配對數據訓練
- **ResNet生成器**: 9個殘差塊的深度網路
- **PatchGAN鑑別器**: 高效的patch級別判別
- **GPU支持**: 自動檢測並使用GPU加速訓練
- **循環一致性**: 確保轉換的雙向一致性

## 文件結構

```
dcp/
├── dataset.py              # 數據加載器
├── generator.py            # ResNet生成器
├── discriminator.py        # PatchGAN鑑別器
├── cyclegan_trainer.py     # 訓練邏輯
├── train_cyclegan.py       # 主訓練腳本
├── inference.py            # 推理腳本
├── clean/                  # 清晰圖像資料夾
├── origin/                 # 有霧圖像資料夾
│   ├── Dense_Fog/         # 濃霧圖像
│   ├── Medium_Fog/        # 中等霧圖像
│   └── No_Fog/            # 無霧圖像
├── checkpoints/           # 模型檢查點（訓練後生成）
└── results/               # 訓練結果圖像（訓練後生成）
```

## 環境需求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- PIL (Pillow)
- numpy
- matplotlib
- tqdm

## 使用說明

### 1. 訓練模型

```bash
python train_cyclegan.py
```

**主要參數說明**:
- `batch_size`: 批次大小 (默認: 4)
- `image_size`: 圖像大小 (默認: 256)
- `n_epochs`: 訓練輪數 (默認: 200)
- `lr`: 學習率 (默認: 0.0002)
- `lambda_cycle`: 循環一致性損失權重 (默認: 10.0)

### 2. 推理測試

```bash
# 處理單張圖像
python inference.py --model_path checkpoints/defogging_generator.pth --input_dir test_image.jpg --output_dir results

# 批量處理
python inference.py --model_path checkpoints/defogging_generator.pth --input_dir test_images/ --output_dir results
```

### 3. 模型架構詳情

#### ResNet生成器
- 輸入: 3通道RGB圖像 (256x256)
- 架構: 卷積 -> 下採樣 -> 9個殘差塊 -> 上採樣 -> 卷積
- 輸出: 3通道RGB圖像 (256x256)
- 激活: ReLU + Tanh (輸出層)
- 正規化: Instance Normalization

#### PatchGAN鑑別器
- 輸入: 3通道RGB圖像 (256x256)
- 架構: 多層卷積網路
- 輸出: 30x30 特徵圖 (每個元素對應輸入的一個patch)
- 激活: LeakyReLU
- 正規化: Instance Normalization

### 4. 損失函數

- **對抗損失**: 使生成圖像看起來真實
- **循環一致性損失**: A→B→A 和 B→A→B 的重構誤差
- **身份損失**: 保持相同域圖像不變 (可選)

總損失 = GAN損失 + λ₁×循環損失 + λ₂×身份損失

### 5. 訓練提示

1. **數據平衡**: 確保清晰圖像和有霧圖像數量相近
2. **GPU記憶體**: 如果出現記憶體不足，減少batch_size
3. **訓練時間**: 完整訓練需要數小時到數天（取決於數據量和硬體）
4. **檢查點**: 每10個epoch自動保存，可隨時恢復訓練
5. **監控**: 觀察損失曲線，確保生成器和鑑別器平衡

### 6. 結果評估

- 查看 `results/` 資料夾中的訓練過程圖像
- 循環一致性損失應該逐漸降低
- 生成圖像質量應該逐步改善
- 可以使用PSNR、SSIM等指標量化評估

### 7. 故障排除

**常見問題**:

1. **CUDA記憶體不足**:
   ```python
   # 在train_cyclegan.py中調整
   config['batch_size'] = 2  # 減少批次大小
   ```

2. **訓練不穩定**:
   ```python
   # 調整學習率和損失權重
   config['lr'] = 0.0001
   config['lambda_cycle'] = 5.0
   ```

3. **生成質量差**:
   - 增加訓練輪數
   - 檢查數據質量
   - 調整網路架構參數

4. **模式崩潰**:
   - 檢查鑑別器是否過強
   - 調整學習率比例
   - 使用不同的初始化

## 高級配置

如需自定義訓練參數，編輯 `train_cyclegan.py` 中的 `config` 字典：

```python
config = {
    'batch_size': 4,           # 批次大小
    'image_size': 256,         # 圖像尺寸
    'lr': 0.0002,             # 學習率
    'n_epochs': 200,          # 總訓練輪數
    'n_epochs_decay': 100,    # 學習率衰減輪數
    'lambda_cycle': 10.0,     # 循環一致性權重
    'lambda_identity': 0.5,   # 身份映射權重
    'n_residual_blocks': 9,   # 殘差塊數量
    'ngf': 64,               # 生成器特徵數
    'ndf': 64,               # 鑑別器特徵數
}
```

## 參考文獻

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

## 授權

本項目僅供學習和研究使用。