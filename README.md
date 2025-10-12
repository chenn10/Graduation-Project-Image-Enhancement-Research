# CycleGAN 去霧系統

這是一個基於CycleGAN的圖像去霧系統，能夠將有霧圖像轉換為清晰圖像。

## 系統特點

- **CycleGAN架構**: 使用無配對數據訓練
- **ResNet生成器**: 9個殘差塊的深度網路
- **PatchGAN鑑別器**: 高效的patch級別判別
- **GPU支持**: 自動檢測並使用GPU加速訓練
- **循環一致性**: 確保轉換的雙向一致性

## 環境需求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- PIL (Pillow)
- numpy
- matplotlib
- tqdm
##

模型請至https://drive.google.com/drive/folders/1BsrbmEJioZPdssDrPnVXkEu1II3KQxY3?usp=sharing進行下載

清晰圖像訓練集:https://www.kaggle.com/datasets/klemenko/kitti-dataset

有霧圖像訓練集:https://www.kaggle.com/datasets/yessicatuteja/foggy-cityscapes-image-dataset

使用者須自行建立兩個空資料夾分別是:uploads以及web_result用來儲存在網站上傳的照片 以及 進行去霧後的結果

執行時請執行以下代碼:cd /path/to/yourproject/python v6_multi_model_web.py
便可以在瀏覽器中開啟:http://localhost:5000使用環境

##
本專題尚在持續優化中，若有更新將會上傳於此



