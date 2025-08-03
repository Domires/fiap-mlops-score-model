@echo off
echo ===============================================
echo  Ativando ambiente virtual do projeto MLOps
echo ===============================================

REM Criar venv se não existir
if not exist "venv" (
    echo Criando ambiente virtual...
    python -m venv venv
    echo Ambiente virtual criado!
)

REM Ativar venv
echo Ativando ambiente virtual...
call .\venv\Scripts\activate.bat

REM Verificar se está ativo
echo.
echo ✅ Ambiente virtual ativado!
echo Python: %VIRTUAL_ENV%
echo.
echo Para instalar dependências execute:
echo pip install -r requirements.txt
echo.
echo Para testar o modelo execute:
echo python train_credit_score_model.py
echo.
echo ===============================================