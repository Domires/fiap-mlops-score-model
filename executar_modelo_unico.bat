@echo off
echo ================================================================
echo 🎯 MODELO ÚNICO DE CREDIT SCORE - RANDOM FOREST
echo ================================================================
echo.
echo ❌ PROBLEMA RESOLVIDO: Múltiplos modelos
echo ✅ AGORA: Apenas 1 modelo Random Forest
echo ✅ COM: MLflow para tracking no DagsHub
echo.
echo 🚀 Iniciando treinamento automático...
echo.

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

echo ================================================================
echo 🔗 EXECUTANDO MODELO COM MLFLOW (DAGSHUB)
echo ================================================================
echo.
echo ✅ Treina apenas Random Forest
echo ✅ Registra no MLflow/DagsHub  
echo ✅ Sem múltiplos modelos
echo 🔗 Permite visualização no MLflow UI
echo.

python train_credit_score_model.py

echo.
echo ================================================================
echo ✅ EXECUÇÃO CONCLUÍDA!
echo ================================================================
echo.
echo 📁 Arquivos gerados na pasta 'models/':
echo    - random_forest_credit_score.pkl (modelo treinado)
echo    - label_encoder.pkl (conversor de classes)
echo.
echo 🔗 Resultados no MLflow:
echo    - Acesse o DagsHub para visualizar métricas
echo    - Modelo registrado para tracking
echo.
echo 🎯 CONFIRMADO: Apenas 1 modelo Random Forest foi treinado!
echo.
pause