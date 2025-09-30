@echo off
REM 系統設置腳本
echo ========================================
echo CycleGAN 去霧系統環境設置
echo ========================================
echo.

REM 檢查Python是否安裝
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 錯誤: 找不到Python，請先安裝Python 3.8+
    pause
    exit /b 1
)

REM 創建虛擬環境
if not exist ".venv" (
    echo 創建虛擬環境...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo 虛擬環境創建失敗
        pause
        exit /b 1
    )
)

REM 激活虛擬環境並安裝依賴
echo 安裝依賴套件...
.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\pip.exe install pillow numpy matplotlib tqdm

REM 創建必要資料夾
if not exist "checkpoints" mkdir checkpoints
if not exist "results" mkdir results
if not exist "defogged_results" mkdir defogged_results

echo.
echo 設置完成！
echo 現在可以運行 run.bat 來使用系統
echo.
pause