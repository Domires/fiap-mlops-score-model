@echo off
echo ================================================================
echo 🎯 MODELO ÚNICO DE CREDIT SCORE - RANDOM FOREST
echo ================================================================
echo.
echo ❌ PROBLEMA RESOLVIDO: Múltiplos modelos
echo ✅ AGORA: Apenas 1 modelo Random Forest
echo ✅ Duas opções: COM ou SEM MLflow
echo.
echo Escolha uma opção:
echo.
echo 1) Script Simples (SEM MLflow) - simple_credit_score_model.py
echo 2) Script com MLflow (DagsHub) - train_credit_score_model.py
echo 3) Sair
echo.
set /p choice="Digite sua escolha (1, 2 ou 3): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Executando Script Simples (RECOMENDADO)...
    goto :script_simples
)

if "%choice%"=="2" (
    echo.
    echo 🔧 Executando Script Original Modificado...
    goto :script_original
)

if "%choice%"=="3" (
    echo.
    echo 👋 Saindo...
    exit /b 0
)

echo.
echo ❌ Opção inválida. Execute novamente.
pause
exit /b 1

:script_simples
echo.
echo ================================================================
echo 🚀 EXECUTANDO SCRIPT SIMPLES - SEM MLFLOW
echo ================================================================
echo.
echo ✅ Mais robusto e completo
echo ✅ Visualizações automáticas
echo ✅ SEM MLflow (evita problemas de endpoint)
echo ✅ Salvamento local apenas
echo.

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

python simple_credit_score_model.py
goto :resultado

:script_original
echo.
echo ================================================================
echo 🔗 EXECUTANDO SCRIPT COM MLFLOW (DAGSHUB)
echo ================================================================
echo.
echo ✅ Treina apenas Random Forest
echo ✅ Registra no MLflow/DagsHub
echo ✅ Sem múltiplos modelos
echo 🔗 Permite visualização no MLflow UI
echo.

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

python train_credit_score_model.py
goto :resultado

:resultado
echo.
echo ================================================================
echo ✅ EXECUÇÃO CONCLUÍDA!
echo ================================================================
echo.
echo 📁 Verifique os resultados na pasta 'models/':
echo    - random_forest_credit_score.pkl (modelo treinado)
echo    - label_encoder.pkl (conversor de classes)
echo    - predictions.csv (predições)
echo    - model_info.json (informações do modelo)
echo    - confusion_matrix.png (matriz de confusão - se script simples)
echo    - feature_importance.png (importância das features - se script simples)
echo.
echo 🎯 CONFIRMADO: Apenas 1 modelo Random Forest foi treinado!
echo.
echo 💡 DIFERENÇA ENTRE AS OPÇÕES:
echo    Opção 1: Mais robusta, sem MLflow, visualizações automáticas
echo    Opção 2: Registra no MLflow/DagsHub para tracking de experimentos
echo.
pause