@echo off
chcp 65001 >nul 2>&1
echo.
echo  ShopBR v2 - MLOps Platform
echo  ===========================
echo.

cd /d "%~dp0"

echo [1/5] Instalando dependencias...
pip install -r requirements.txt -q
if errorlevel 1 ( echo ERRO nas dependencias & pause & exit /b 1 )

echo.
echo [2/5] Inicializando banco de dados e gerando dados...
python -c "import sys; sys.path.insert(0,'%~dp0'); from src.data.gerar_dados import gerar; gerar()"
if errorlevel 1 ( echo ERRO nos dados & pause & exit /b 1 )

echo.
echo [3/5] Criando features...
python -c "import sys,os; sys.path.insert(0,'%~dp0'); os.makedirs('%~dp0data/processed',exist_ok=True); from src.features.feature_engineering import criar_features_categoria, criar_features_sku; import pandas as pd; df=criar_features_categoria(); df.to_csv('%~dp0data/processed/features_categoria.csv',index=False); df2=criar_features_sku(); df2.to_csv('%~dp0data/processed/features_sku.csv',index=False); print('Features OK')"
if errorlevel 1 ( echo ERRO no feature engineering & pause & exit /b 1 )

echo.
echo [4/5] Treinando modelos (pode demorar 2-4 min)...
python -c "import sys; sys.path.insert(0,'%~dp0'); from src.models.trainer import treinar; treinar('categoria',forcar_promocao=True)"
if errorlevel 1 ( echo ERRO no treinamento & pause & exit /b 1 )

echo.
echo [5/5] Verificando drift inicial...
python -c "import sys; sys.path.insert(0,'%~dp0'); from src.monitoring.drift_monitor import relatorio_saude; r=relatorio_saude(); print('Saude OK:', list(r.keys()))"

echo.
echo  ========================================
echo   Pipeline concluido!
echo   Abrindo: http://localhost:5050
echo  ========================================
echo.
echo  Pressione CTRL+C para encerrar.
echo.

python dashboard\app.py
pause
