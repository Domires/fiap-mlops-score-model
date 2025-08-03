#!/usr/bin/env python3
"""
⚠️ ARQUIVO ATUALIZADO - AGORA TREINA APENAS 1 MODELO

PROBLEMA RESOLVIDO: Este arquivo foi modificado para treinar APENAS Random Forest.

RECOMENDAÇÃO: Use o arquivo 'simple_credit_score_model.py' que é mais robusto:
    python simple_credit_score_model.py

Este script executa:
1. Carregamento e pré-processamento dos dados
2. Treinamento APENAS do Random Forest
3. Avaliação e registro

Usage:
    python train_credit_score_model.py
"""

import sys
import os

# Verificar se deve usar o script simples
print("⚠️ AVISO: Este arquivo foi modificado para treinar apenas Random Forest.")
print("💡 RECOMENDAÇÃO: Use 'python simple_credit_score_model.py' para melhor experiência.")
print("   Pressione Enter para continuar ou Ctrl+C para cancelar...")

try:
    input()
except KeyboardInterrupt:
    print("\n🚀 Execute: python simple_credit_score_model.py")
    sys.exit(0)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import main_random_forest_mlflow

if __name__ == "__main__":
    print("=== INICIANDO TREINAMENTO ÚNICO - RANDOM FOREST COM MLFLOW ===")
    print("🎯 APENAS 1 MODELO será treinado (Random Forest)")
    print("✅ COM MLflow (registrado no DagsHub)")
    print("✅ SEM múltiplos modelos\n")
    
    try:
        main_random_forest_mlflow()
        print("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
        print("✅ Apenas 1 modelo Random Forest foi treinado!")
        print("🔗 Verifique o resultado no DagsHub MLflow UI")
    except Exception as e:
        print(f"\n=== ERRO DURANTE O TREINAMENTO ===")
        print(f"Erro: {e}")
        print("\n💡 ALTERNATIVA: Execute 'python simple_credit_score_model.py' (sem MLflow)")
        sys.exit(1)