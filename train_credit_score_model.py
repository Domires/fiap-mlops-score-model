#!/usr/bin/env python3
"""
Script para treinamento do modelo de Credit Score com Random Forest.

MODIFICADO: Treina APENAS 1 modelo Random Forest com MLflow.

Este script executa:
1. Carregamento e pré-processamento dos dados
2. Treinamento do Random Forest
3. Registro no MLflow/DagsHub
4. Avaliação e métricas

Usage:
    python train_credit_score_model.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import main

if __name__ == "__main__":
    print("=== TREINAMENTO DE MODELO DE CREDIT SCORE ===")
    print("🎯 Modelo: Random Forest (único modelo)")
    print("🔗 MLflow: Ativado (DagsHub)")
    print("📊 Métricas: Registradas automaticamente")
    print("⚠️ MODIFICADO: Treina apenas Random Forest (sem múltiplos modelos)\n")
    
    try:
        main()
        print("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
        print("✅ Modelo Random Forest treinado e registrado no MLflow!")
        print("🔗 Acesse o DagsHub para visualizar os resultados")
    except Exception as e:
        print(f"\n=== ERRO DURANTE O TREINAMENTO ===")
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)