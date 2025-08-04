#!/usr/bin/env python3
"""
Script de exemplo para testar o modelo registrado no MLflow
Simula como seria usado em uma API de produção para classificação de Credit Score

Uso:
    python test_model_api.py
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import json
import os

def load_model_info():
    """Carrega informações do modelo salvas durante o treinamento"""
    try:
        with open('models/api_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Arquivo models/api_info.json não encontrado")
        print("💡 Execute primeiro: python train_credit_score_model.py")
        return None

def load_input_example():
    """Carrega exemplo de entrada salvo durante o treinamento"""
    try:
        return pd.read_csv('models/input_example.csv')
    except FileNotFoundError:
        print("❌ Arquivo models/input_example.csv não encontrado")
        return None

def test_model_prediction():
    """Testa o modelo registrado com dados de exemplo"""
    
    print("=" * 60)
    print("🔌 TESTE DE API - MODELO DE CREDIT SCORE")
    print("=" * 60)
    
    # Configurar MLflow
    import dagshub
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/domires/fiap-mlops-score-model.mlflow")
    
    # Carregar informações do modelo
    model_info = load_model_info()
    if not model_info:
        return
    
    # Carregar exemplo de entrada
    input_example = load_input_example()
    if input_example is None:
        return
    
    print("📊 INFORMAÇÕES DO MODELO:")
    print(f"   🔗 Nome: {model_info['model_name']}")
    print(f"   📊 Versão: {model_info['model_version']}")
    print(f"   📊 Run ID: {model_info['run_id']}")
    print(f"   📊 Features: {model_info['feature_count']}")
    print(f"   📊 Classes: {model_info['classes']}")
    print()
    
    try:
        # Carregar modelo do MLflow Model Registry
        model_uri = f"models:/{model_info['model_name']}/{model_info['model_version']}"
        print(f"🚀 Carregando modelo: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modelo carregado com sucesso!")
        print()
        
        # Fazer predições com dados de exemplo
        print("📊 DADOS DE ENTRADA (3 amostras):")
        print(input_example)
        print()
        
        print("🚀 Fazendo predições...")
        predictions = model.predict(input_example)
        
        print("✅ RESULTADOS:")
        for i, pred in enumerate(predictions):
            print(f"   Cliente {i+1}: {pred}")
        print()
        
        # Testar predição com um único cliente (simular requisição API)
        print("🔌 SIMULANDO REQUISIÇÃO DE API (1 cliente):")
        single_client = input_example.iloc[[0]]  # Primeiro cliente
        single_prediction = model.predict(single_client)
        
        print("📊 INPUT (JSON format para API):")
        client_json = single_client.to_dict('records')[0]
        print(json.dumps(client_json, indent=2))
        print()
        
        print("📊 OUTPUT:")
        print(f"   Credit Score: {single_prediction[0]}")
        print()
        
        # Testar probabilidades se disponível
        try:
            print("🎯 Testando probabilidades...")
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(single_client)
                print("✅ PROBABILIDADES:")
                for i, prob in enumerate(probabilities[0]):
                    class_name = model_info['classes'][i] if i < len(model_info['classes']) else f"Classe_{i}"
                    print(f"   {class_name}: {prob:.3f}")
            else:
                print("⚠️ predict_proba não disponível neste modelo")
        except Exception as prob_error:
            print(f"⚠️ Erro ao calcular probabilidades: {prob_error}")
        
        print()
        print("=" * 60)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print("🔌 O modelo está pronto para uso em API de produção!")
        print()
        print("📋 EXEMPLO DE USO EM API:")
        print("   1. Receber dados do cliente em JSON")
        print("   2. Converter para DataFrame")
        print("   3. Chamar model.predict(dados)")
        print("   4. Retornar resultado da classificação")
        print()
        print("🎯 PRÓXIMOS PASSOS:")
        print("   - Implementar endpoint REST API")
        print("   - Adicionar validação de entrada")
        print("   - Configurar monitoramento do modelo")
        print("   - Implementar logs de predições")
        
    except Exception as e:
        print(f"❌ ERRO AO TESTAR MODELO: {e}")
        print("💡 Verifique se o modelo foi registrado corretamente")
        print("💡 Execute: python train_credit_score_model.py")

if __name__ == "__main__":
    test_model_prediction()