#!/usr/bin/env python3
"""
Código correto para registrar modelo no notebook
Corrige os problemas identificados: path e configuração MLflow
"""

import mlflow
import dagshub

# 1. CONFIGURAR MLFLOW (CRUCIAL!)
print("🔧 Configurando MLflow para DagsHub...")
dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
tracking_uri = "https://dagshub.com/domires/fiap-mlops-score-model.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
print(f"✅ Registry URI: {mlflow.get_registry_uri()}")

# 2. DEFINIR DADOS (CORRETOS)
run_id = "1d7d92097a124cd286d341a369291aa2"  # Seu run_id atual
model_name = "fiap-mlops-score-model"        # Nome atualizado

# 3. PATH CORRETO DO MODELO
# ❌ ERRADO: f"runs:/{run_id}/model"
# ✅ CORRETO: f"runs:/{run_id}/random_forest_model"
model_uri = f"runs:/{run_id}/random_forest_model"

print(f"📊 Run ID: {run_id}")
print(f"🎯 Nome do modelo: {model_name}")
print(f"🔗 Model URI: {model_uri}")

# 4. REGISTRAR MODELO
try:
    print("\n🚀 Registrando modelo...")
    registered_model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print("✅ SUCESSO!")
    print(f"🔗 Modelo: {model_name}")
    print(f"📊 Versão: {registered_model_version.version}")
    print(f"📊 Run ID: {run_id}")
    print("\n🎯 Verifique na aba 'Models' do MLflow UI!")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    print("\n💡 POSSÍVEIS CAUSAS:")
    print("   1. Run ID não existe")
    print("   2. Path do modelo incorreto")
    print("   3. MLflow não configurado")
    print("   4. Permissões do DagsHub")