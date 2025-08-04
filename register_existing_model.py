#!/usr/bin/env python3
"""
Script para registrar modelo existente no MLflow Model Registry
Conforme documentação do curso MLOPS

Para registrar precisamos do "run_id" do experimento que será promovido ao modelo.
A cada novo registro uma nova versão será gerada.
"""

import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import register_existing_model

if __name__ == "__main__":
    print("=" * 60)
    print("🔗 REGISTRO DE MODELO EXISTENTE NO MLFLOW")
    print("=" * 60)
    print("📋 Conforme documentação do curso MLOPS")
    print("🎯 Usando mlflow.register_model() com run_id específico")
    print()
    
    # Seu run_id específico (ATUALIZADO)
    your_run_id = "1d7d92097a124cd286d341a369291aa2"  # ← Novo run_id do DagsHub
    model_name = "fiap-mlops-score-model"
    
    print(f"📊 Run ID: {your_run_id}")
    print(f"🎯 Nome do modelo: {model_name}")
    print()
    
    try:
        # Registrar o modelo existente
        registered_version = register_existing_model(
            run_id=your_run_id,
            model_name=model_name
        )
        
        if registered_version:
            print("\n" + "=" * 60)
            print("✅ MODELO REGISTRADO COM SUCESSO!")
            print("=" * 60)
            print(f"🔗 Nome: {model_name}")
            print(f"📊 Versão: {registered_version.version}")
            print(f"📊 Run ID: {your_run_id}")
            print()
            print("🎯 PRÓXIMOS PASSOS:")
            print("   1. Acesse o MLflow UI no DagsHub")
            print("   2. Vá para a aba 'Models'")
            print(f"   3. Procure por '{model_name}'")
            print(f"   4. Veja a versão {registered_version.version} registrada")
            print()
            print("📋 CONFORME DOCUMENTAÇÃO:")
            print("   - A cada novo registro uma nova versão será gerada")
            print("   - Use mlflow.register_model() para promover experimentos")
        else:
            print("\n❌ FALHA NO REGISTRO")
            print("💡 Verifique se o run_id existe e contém o modelo")
            
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O REGISTRO")
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)