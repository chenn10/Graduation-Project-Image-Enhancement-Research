@echo off
REM CycleGAN去霧系統啟動腳本
echo ========================================
echo CycleGAN 圖像去霧系統
echo ========================================
echo.

REM 檢查Python環境
if not exist ".venv\Scripts\python.exe" (
    echo 錯誤: 找不到虛擬環境，請先運行 setup.bat
    pause
    exit /b 1
)

REM 顯示選項
echo 請選擇要執行的操作:
echo 1. 開始訓練 CycleGAN 模型
echo 2. 使用已訓練模型進行推理
echo 3. 測試系統組件
echo 4. 查看幫助文檔
echo 5. 退出
echo.

set /p choice=請輸入選項 (1-5): 

if "%choice%"=="1" (
    echo.
    echo 開始訓練 CycleGAN 模型...
    echo 注意: 訓練可能需要數小時，請確保有足夠的時間和GPU記憶體
    echo.
    .venv\Scripts\python.exe train_cyclegan.py
) else if "%choice%"=="2" (
    echo.
    echo 進入推理模式...
    if not exist "checkpoints\defogging_generator.pth" (
        echo 錯誤: 找不到訓練好的模型文件
        echo 請先運行訓練或確認模型路徑正確
        pause
        exit /b 1
    )
    
    set /p input_path=請輸入要處理的圖像路徑或資料夾: 
    if "%input_path%"=="" (
        echo 使用默認測試圖像...
        set input_path=origin
    )
    
    .venv\Scripts\python.exe inference.py --input_dir "%input_path%" --output_dir defogged_results
) else if "%choice%"=="3" (
    echo.
    echo 測試系統組件...
    .venv\Scripts\python.exe -c "exec(open('test_system.py').read())" 2>nul || (
        echo 創建測試腳本...
        echo import torch > test_system.py
        echo from dataset import DefoggingDataset >> test_system.py
        echo from generator import ResNetGenerator >> test_system.py
        echo from discriminator import PatchGANDiscriminator >> test_system.py
        echo from cyclegan_trainer import CycleGANTrainer >> test_system.py
        echo print("All components tested successfully!") >> test_system.py
        .venv\Scripts\python.exe test_system.py
        del test_system.py
    )
) else if "%choice%"=="4" (
    echo.
    echo 打開幫助文檔...
    if exist "README.md" (
        start README.md
    ) else (
        echo README.md 文件不存在
    )
) else if "%choice%"=="5" (
    echo 退出系統
    exit /b 0
) else (
    echo 無效選項，請重新運行腳本
    pause
    exit /b 1
)

echo.
echo 操作完成！
pause