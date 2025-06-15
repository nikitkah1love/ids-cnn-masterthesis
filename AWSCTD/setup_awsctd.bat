@echo off
echo [INFO] Перевірка наявності Python 3.10...

where python3.10 >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10 не знайдено. Завантаж тут: https://www.python.org/ftp/python/3.10.12/python-3.10.12-amd64.exe
    pause
    exit /b 1
)

echo [INFO] Створення віртуального середовища "venv"
python3.10 -m venv venv

echo [INFO] Активація середовища
call venv\Scripts\activate.bat

echo [INFO] Оновлення pip
python -m pip install --upgrade pip

echo [INFO] Створення requirements.txt
echo tensorflow==2.12.0> requirements.txt
echo numpy==1.22.0>> requirements.txt
echo pandas==1.4.0>> requirements.txt
echo scikit-learn==1.0.2>> requirements.txt
echo matplotlib==3.5.0>> requirements.txt
echo scipy==1.8.0>> requirements.txt

echo [INFO] Встановлення залежностей
pip install -r requirements.txt

echo [INFO] Все готово!
echo =========================================
echo Для запуску моделі:
echo 1. Активуй середовище:
echo    call venv\Scripts\activate.bat
echo 2. Перейди у теку Python:
echo    cd Python
echo 3. Запусти:
echo    python AWSCTD.py MalwarePlusClean\1000_2.csv AWSCTD-CNN-S
echo =========================================
pause

