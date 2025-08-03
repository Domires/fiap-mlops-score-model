@echo off
echo ================================================================
echo 🎯 EXECUTANDO MODELO ÚNICO DE CREDIT SCORE - RANDOM FOREST
echo ================================================================
echo.
echo ✅ APENAS 1 MODELO será treinado (Random Forest)
echo ✅ SEM MLflow (evita problemas de endpoint)
echo ✅ SEM múltiplos modelos
echo.
echo Iniciando treinamento...
echo.

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

REM Executar o script Python
echo 🚀 Executando modelo único...
python simple_credit_score_model.py

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
echo    - confusion_matrix.png (matriz de confusão)
echo    - feature_importance.png (importância das features)
echo.
pause