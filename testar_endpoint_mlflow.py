#!/usr/bin/env python3
"""
Script para testar o endpoint do modelo no MLflow
Verifica se o modelo foi registrado corretamente e está acessível
"""

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import os

def testar_endpoint_mlflow():
    """Testa se o modelo registrado no MLflow está funcionando"""
    
    print("=" * 60)
    print("🔌 TESTE DO ENDPOINT DO MODELO NO MLFLOW")
    print("=" * 60)
    print("🎯 Verificando se o modelo está acessível e funcionando")
    print()
    
    # Configurar MLflow
    print("🔧 Configurando conexão com DagsHub...")
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/domires/fiap-mlops-score-model.mlflow")
    
    print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
    print()
    
    # IDs dos runs que criamos
    run_ids = [
        "2f5087600685403383420bf1c6720ed5",  # Último run (padrão FIAP)
        "bcadaadae75c4ea499bcdad78e9a1d11"   # Run anterior (registro básico)
    ]
    
    for i, run_id in enumerate(run_ids, 1):
        print(f"🚀 TESTE {i}: Run ID {run_id}")
        print("-" * 50)
        
        try:
            # Tentar diferentes formas de carregar o modelo
            model_uris = [
                f"runs:/{run_id}/model",
                f"runs:/{run_id}/random_forest_credit_score.pkl"
            ]
            
            modelo_carregado = None
            
            for model_uri in model_uris:
                try:
                    print(f"📊 Tentando carregar: {model_uri}")
                    
                    # Tentar carregar como pyfunc primeiro
                    try:
                        modelo_carregado = mlflow.pyfunc.load_model(model_uri)
                        print("✅ Modelo carregado como pyfunc!")
                        break
                    except Exception as pyfunc_error:
                        print(f"⚠️ pyfunc falhou: {pyfunc_error}")
                        
                        # Tentar carregar como sklearn
                        try:
                            modelo_carregado = mlflow.sklearn.load_model(model_uri)
                            print("✅ Modelo carregado como sklearn!")
                            break
                        except Exception as sklearn_error:
                            print(f"⚠️ sklearn falhou: {sklearn_error}")
                            continue
                            
                except Exception as load_error:
                    print(f"❌ Erro ao carregar {model_uri}: {load_error}")
                    continue
            
            if modelo_carregado is None:
                print("❌ Não foi possível carregar o modelo deste run")
                continue
            
            # Criar dados de teste
            print("\n📊 Criando dados de teste...")
            
            # Dados de exemplo (baseado no que esperamos do modelo de credit score)
            dados_teste = pd.DataFrame({
                'Age': [25, 35, 45],
                'Annual_Income': [50000, 75000, 100000], 
                'Monthly_Inhand_Salary': [4000, 6000, 8000],
                'Num_Bank_Accounts': [2, 3, 4],
                'Num_Credit_Card': [1, 2, 3],
                'Interest_Rate': [15, 12, 10],
                'Num_of_Loan': [1, 2, 0],
                'Delay_from_due_date': [0, 5, 0],
                'Num_of_Delayed_Payment': [0, 1, 0],
                'Changed_Credit_Limit': [0, 5, 10],
                'Num_Credit_Inquiries': [1, 2, 1],
                'Outstanding_Debt': [1000, 2000, 500],
                'Credit_Utilization_Ratio': [0.3, 0.5, 0.2],
                'Total_EMI_per_month': [500, 800, 600],
                'Amount_invested_monthly': [200, 500, 1000],
                'Monthly_Balance': [1000, 2000, 3000]
            })
            
            # Adicionar features categóricas básicas se necessário
            categorical_features = {
                'Month': 'January',
                'Occupation': 'Engineer', 
                'Type_of_Loan': 'Personal Loan',
                'Credit_Mix': 'Standard',
                'Credit_History_Age': '5 Years',
                'Payment_of_Min_Amount': 'Yes',
                'Payment_Behaviour': 'High_spent_Small_value_payments'
            }
            
            for col, value in categorical_features.items():
                if col not in dados_teste.columns:
                    dados_teste[col] = value
            
            print(f"✅ Dados de teste criados: {dados_teste.shape}")
            print(f"📊 Colunas: {list(dados_teste.columns)}")
            
            # Testar predição
            print("\n🔮 Testando predição...")
            try:
                predicoes = modelo_carregado.predict(dados_teste)
                print("✅ PREDIÇÃO FUNCIONOU!")
                print(f"📊 Predições: {predicoes}")
                print(f"📊 Tipo: {type(predicoes)}")
                print(f"📊 Shape: {predicoes.shape if hasattr(predicoes, 'shape') else len(predicoes)}")
                
                # Mostrar resultados
                print("\n📋 RESULTADOS:")
                for i, pred in enumerate(predicoes):
                    print(f"   Cliente {i+1}: {pred}")
                
                # Testar probabilidades se disponível
                try:
                    if hasattr(modelo_carregado, 'predict_proba'):
                        probas = modelo_carregado.predict_proba(dados_teste)
                        print(f"\n🎯 Probabilidades disponíveis: {probas.shape}")
                        print(f"📊 Exemplo probabilidades cliente 1: {probas[0]}")
                except Exception as proba_error:
                    print(f"⚠️ predict_proba não disponível: {proba_error}")
                
                print(f"\n✅ MODELO DO RUN {run_id} ESTÁ FUNCIONANDO PERFEITAMENTE!")
                print(f"🔌 Model URI funcionando: {model_uri}")
                print()
                
                # Se chegou até aqui, o teste foi bem-sucedido
                break
                
            except Exception as pred_error:
                print(f"❌ Erro na predição: {pred_error}")
                print("💡 Possível incompatibilidade de features")
                
                # Tentar com dados mais simples
                try:
                    dados_simples = dados_teste.iloc[:, :5]  # Apenas primeiras 5 colunas
                    print(f"\n🔄 Tentando com dados simplificados: {dados_simples.shape}")
                    predicoes_simples = modelo_carregado.predict(dados_simples)
                    print("✅ PREDIÇÃO COM DADOS SIMPLES FUNCIONOU!")
                    print(f"📊 Predições: {predicoes_simples}")
                except Exception as simple_error:
                    print(f"❌ Erro mesmo com dados simples: {simple_error}")
                
        except Exception as run_error:
            print(f"❌ Erro geral para run {run_id}: {run_error}")
        
        print("\n" + "=" * 50)
    
    print("\n🏁 TESTE CONCLUÍDO!")
    print()
    print("📋 RESUMO:")
    print("   - Se apareceu 'MODELO ESTÁ FUNCIONANDO PERFEITAMENTE'")
    print("   - Então o endpoint está OK e o modelo pode ser usado")
    print("   - O model_uri mostrado é o que deve ser usado em produção")
    print()
    print("🔌 COMO USAR EM PRODUÇÃO:")
    print("   import mlflow.pyfunc")
    print("   model = mlflow.pyfunc.load_model('runs:/RUN_ID/model')")
    print("   predictions = model.predict(data)")

if __name__ == "__main__":
    testar_endpoint_mlflow()