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

from src.model_training import main, register_existing_model

if __name__ == "__main__":
    print("TREINAMENTO DE MODELO DE CREDIT SCORE")
    
    try:
        main()  # Treinar novo modelo
        print("\nTREINAMENTO CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        print(f"\nERRO DURANTE O TREINAMENTO")
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)