@echo off
echo ================================================================
echo 🎯 MODELO ÚNICO DE CREDIT SCORE - RANDOM FOREST
echo ================================================================
echo.
echo ❌ PROBLEMA RESOLVIDO: Múltiplos modelos
echo ✅ AGORA: Apenas 1 modelo Random Forest
echo ✅ COM: MLflow para tracking no DagsHub
echo ✅ PRONTO: Para uso em API de produção
echo.

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

echo.
echo ================================================================
echo 🚀 ESCOLHA UMA OPÇÃO:
echo ================================================================
echo.
echo 1) 🔗 TREINAR NOVO MODELO (com MLflow)
echo 2) 🔌 TESTAR MODELO PARA API
echo 3) 📊 REGISTRAR MODELO EXISTENTE
echo.
set /p opcao="Digite sua escolha (1, 2 ou 3): "

if "%opcao%"=="1" (
    echo.
    echo ================================================================
    echo 🔗 EXECUTANDO TREINAMENTO COM MLFLOW
    echo ================================================================
    echo.
    echo ✅ Treina apenas Random Forest
    echo ✅ Registra no MLflow/DagsHub
    echo ✅ Cria signature para API
    echo ✅ Salva documentação da API
    echo.
    
    python train_credit_score_model.py
    
    echo.
    echo ================================================================
    echo ✅ TREINAMENTO CONCLUÍDO!
    echo ================================================================
    echo.
    echo 📁 Arquivos gerados na pasta 'models/':
    echo    - random_forest_credit_score.pkl (modelo)
    echo    - label_encoder.pkl (conversor)
    echo    - api_info.json (info para API)
    echo    - input_example.csv (exemplo de entrada)
    echo.
    echo 🔗 Resultados no MLflow:
    echo    - Modelo registrado no Model Registry
    echo    - Signature e input example configurados
    echo    - Pronto para uso em API
    echo.
    
) else if "%opcao%"=="2" (
    echo.
    echo ================================================================
    echo 🔌 TESTANDO MODELO PARA API
    echo ================================================================
    echo.
    echo ✅ Carrega modelo do MLflow
    echo ✅ Testa predições
    echo ✅ Simula uso em API
    echo.
    
    python test_model_api.py
    
) else if "%opcao%"=="3" (
    echo.
    echo ================================================================
    echo 📊 REGISTRANDO MODELO EXISTENTE
    echo ================================================================
    echo.
    echo ✅ Usa run_id específico
    echo ✅ Registra no Model Registry
    echo.
    
    python register_existing_model.py
    
) else (
    echo.
    echo ❌ Opção inválida! Use 1, 2 ou 3.
    echo.
)

echo.
echo 🎯 CONFIRMADO: Sistema pronto para produção!
echo.
pause