@echo off
echo ========================================
echo   CycleGAN 智能去霧 Web 系統啟動
echo ========================================
echo.

REM 檢查虛擬環境
if not exist ".venv" (
    echo 錯誤: 找不到虛擬環境，請先運行 setup.bat
    pause
    exit /b 1
)

REM 檢查模型文件
if not exist "checkpoints\cyclegan_epoch_45.pth" (
    echo 警告: 找不到45epoch模型文件
    if not exist "checkpoints\defogging_generator.pth" (
        echo 錯誤: 找不到任何訓練好的模型
        echo 請先運行訓練或確認模型文件存在
        pause
        exit /b 1
    )
)

echo 檢查依賴套件...
.venv\Scripts\pip.exe show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo 安裝Web依賴套件...
    .venv\Scripts\pip.exe install flask opencv-python
)

echo.
echo ✅ 準備就緒！
echo 🌐 正在啟動Web服務器...
echo 📱 請在瀏覽器中打開: http://localhost:5000
echo 🛑 按 Ctrl+C 停止服務
echo.

REM 啟動Flask應用
.venv\Scripts\python.exe web_app.py

echo.
echo Web服務已停止
pause