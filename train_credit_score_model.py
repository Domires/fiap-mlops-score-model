#!/usr/bin/env python3
"""
Script principal para treinamento do modelo de classificação de Credit Score.

Este script executa todo o pipeline de treinamento:
1. Carregamento e pré-processamento dos dados
2. Treinamento de múltiplos modelos
3. Avaliação e registro no MLflow
4. Comparação de resultados

Usage:
    python train_credit_score_model.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import main

if __name__ == "__main__":
    print("=== INICIANDO TREINAMENTO DE MODELOS DE CREDIT SCORE ===")
    print("Este script irá treinar múltiplos modelos de classificação para score de crédito.")
    print("Os resultados serão registrados no MLflow para comparação.\n")
    
    try:
        main()
        print("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
        print("Acesse o MLflow UI para visualizar e comparar os resultados dos modelos.")
    except Exception as e:
        print(f"\n=== ERRO DURANTE O TREINAMENTO ===")
        print(f"Erro: {e}")
        sys.exit(1)