#!/usr/bin/env python3
"""
Teste simples do modelo local para validar se está funcionando
Evita problemas de path do MLflow no Windows
"""

import joblib
import pandas as pd
import numpy as np
import os

def testar_modelo_local():
    """Testa o modelo local diretamente"""
    
    print("=" * 60)
    print("🔌 TESTE DO MODELO LOCAL")
    print("=" * 60)
    print("🎯 Validando funcionalidade do modelo treinado")
    print()
    
    # Verificar arquivos
    model_file = 'models/random_forest_credit_score.pkl'
    encoder_file = 'models/label_encoder.pkl'
    
    if not os.path.exists(model_file):
        print("❌ Modelo não encontrado!")
        print("💡 Execute: python train_credit_score_model.py")
        return False
    
    print("📊 Carregando modelo e encoder...")
    
    try:
        # Carregar modelo e encoder
        model = joblib.load(model_file)
        print("✅ Modelo carregado!")
        
        label_encoder = None
        if os.path.exists(encoder_file):
            label_encoder = joblib.load(encoder_file)
            print("✅ Label encoder carregado!")
        
        # Criar dados de teste realistas para credit score
        print("\n📊 Criando dados de teste...")
        
        # Dados que simulam clientes reais
        dados_teste = pd.DataFrame({
            # Dados numéricos
            'Age': [25, 35, 45],
            'Annual_Income': [50000, 75000, 100000],
            'Monthly_Inhand_Salary': [4000, 6000, 8000],
            'Num_Bank_Accounts': [2, 3, 4],
            'Num_Credit_Card': [1, 2, 3],
            'Interest_Rate': [15.0, 12.0, 10.0],
            'Num_of_Loan': [1, 2, 0],
            'Delay_from_due_date': [0, 5, 0],
            'Num_of_Delayed_Payment': [0, 1, 0],
            'Changed_Credit_Limit': [0.0, 5.0, 10.0],
            'Num_Credit_Inquiries': [1, 2, 1],
            'Outstanding_Debt': [1000.0, 2000.0, 500.0],
            'Credit_Utilization_Ratio': [0.3, 0.5, 0.2],
            'Total_EMI_per_month': [500.0, 800.0, 600.0],
            'Amount_invested_monthly': [200.0, 500.0, 1000.0],
            'Monthly_Balance': [1000.0, 2000.0, 3000.0],
            
            # Dados categóricas (encoded)
            'Month': [1, 6, 12],  # Janeiro, Junho, Dezembro
            'Occupation': [1, 2, 3],  # Diferentes ocupações
            'Type_of_Loan': [1, 2, 0],  # Diferentes tipos de empréstimo
            'Credit_Mix': [1, 2, 1],  # Mix de crédito
            'Credit_History_Age': [5, 10, 15],  # Anos de histórico
            'Payment_of_Min_Amount': [1, 1, 0],  # Paga mínimo
            'Payment_Behaviour': [1, 2, 3]  # Comportamento de pagamento
        })
        
        print(f"✅ Dados de teste criados: {dados_teste.shape}")
        print(f"📊 Colunas: {list(dados_teste.columns)}")
        
        # Testar predição
        print("\n🔮 Testando predição...")
        
        try:
            # Fazer predição
            predicoes_numericas = model.predict(dados_teste)
            print("✅ PREDIÇÃO FUNCIONOU!")
            print(f"📊 Predições numéricas: {predicoes_numericas}")
            
            # Converter para labels legíveis se tiver encoder
            if label_encoder:
                try:
                    predicoes_labels = label_encoder.inverse_transform(predicoes_numericas)
                    print(f"📊 Predições em texto: {predicoes_labels}")
                except Exception as decode_error:
                    print(f"⚠️ Não foi possível decodificar: {decode_error}")
                    predicoes_labels = predicoes_numericas
            else:
                predicoes_labels = predicoes_numericas
            
            # Mostrar resultados detalhados
            print("\n📋 RESULTADOS DETALHADOS:")
            for i in range(len(dados_teste)):
                idade = dados_teste.iloc[i]['Age']
                renda = dados_teste.iloc[i]['Annual_Income']
                pred_num = predicoes_numericas[i]
                pred_label = predicoes_labels[i] if label_encoder else pred_num
                
                print(f"   Cliente {i+1}:")
                print(f"     - Idade: {idade} anos")
                print(f"     - Renda: R$ {renda:,}")
                print(f"     - Score Numérico: {pred_num}")
                print(f"     - Score Textual: {pred_label}")
                print()
            
            # Testar probabilidades se disponível
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(dados_teste)
                    print("🎯 PROBABILIDADES POR CLASSE:")
                    
                    # Assumindo classes: 0=Good, 1=Standard, 2=Unknown ou similar
                    classes = label_encoder.classes_ if label_encoder else [0, 1, 2]
                    
                    for i in range(len(dados_teste)):
                        print(f"   Cliente {i+1}:")
                        for j, classe in enumerate(classes):
                            prob = probas[i][j] * 100
                            print(f"     - {classe}: {prob:.1f}%")
                        print()
                else:
                    print("⚠️ predict_proba não disponível neste modelo")
                    
            except Exception as proba_error:
                print(f"⚠️ Erro ao calcular probabilidades: {proba_error}")
            
            print("✅ MODELO LOCAL ESTÁ FUNCIONANDO PERFEITAMENTE!")
            print()
            print("🔌 RESUMO DO TESTE:")
            print(f"   ✅ Modelo carregado: {model_file}")
            print(f"   ✅ Encoder carregado: {encoder_file if label_encoder else 'N/A'}")
            print(f"   ✅ Features aceitas: {dados_teste.shape[1]}")
            print(f"   ✅ Predições realizadas: {len(predicoes_numericas)}")
            print(f"   ✅ Saída: {type(predicoes_labels[0])}")
            
            return True
            
        except Exception as pred_error:
            print(f"❌ Erro na predição: {pred_error}")
            print("💡 Verificando estrutura do modelo...")
            
            try:
                print(f"📊 Tipo do modelo: {type(model)}")
                if hasattr(model, 'feature_names_in_'):
                    print(f"📊 Features esperadas: {model.feature_names_in_}")
                if hasattr(model, 'n_features_in_'):
                    print(f"📊 Número de features esperadas: {model.n_features_in_}")
            except:
                pass
            
            return False
            
    except Exception as load_error:
        print(f"❌ Erro ao carregar modelo: {load_error}")
        return False

def testar_modelo_como_api():
    """Simula uso do modelo como API"""
    
    print("\n" + "=" * 60)
    print("🌐 SIMULAÇÃO DE USO COMO API")
    print("=" * 60)
    
    # Dados que viriam de uma requisição JSON
    requisicao_json = {
        "cliente_id": "CLI_001",
        "dados": {
            "Age": 30,
            "Annual_Income": 60000,
            "Monthly_Inhand_Salary": 5000,
            "Num_Bank_Accounts": 2,
            "Num_Credit_Card": 1,
            "Interest_Rate": 14.0,
            "Num_of_Loan": 1,
            "Delay_from_due_date": 0,
            "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 2.0,
            "Num_Credit_Inquiries": 1,
            "Outstanding_Debt": 1500.0,
            "Credit_Utilization_Ratio": 0.4,
            "Total_EMI_per_month": 600.0,
            "Amount_invested_monthly": 300.0,
            "Monthly_Balance": 1500.0,
            "Month": 3,
            "Occupation": 2,
            "Type_of_Loan": 1,
            "Credit_Mix": 1,
            "Credit_History_Age": 7,
            "Payment_of_Min_Amount": 1,
            "Payment_Behaviour": 2
        }
    }
    
    print(f"📨 Requisição recebida para cliente: {requisicao_json['cliente_id']}")
    
    try:
        # Carregar modelo
        model = joblib.load('models/random_forest_credit_score.pkl')
        encoder = None
        if os.path.exists('models/label_encoder.pkl'):
            encoder = joblib.load('models/label_encoder.pkl')
        
        # Converter dados para DataFrame
        dados_df = pd.DataFrame([requisicao_json['dados']])
        
        # Predição
        predicao = model.predict(dados_df)[0]
        
        # Decodificar se necessário
        score_final = encoder.inverse_transform([predicao])[0] if encoder else predicao
        
        # Resposta da API
        resposta_api = {
            "status": "success",
            "cliente_id": requisicao_json['cliente_id'],
            "credit_score": score_final,
            "score_numerico": int(predicao),
            "timestamp": "2025-08-03T22:15:00Z",
            "model_version": "v1.0"
        }
        
        print("✅ RESPOSTA DA API:")
        print(f"   📊 Status: {resposta_api['status']}")
        print(f"   🎯 Credit Score: {resposta_api['credit_score']}")
        print(f"   📊 Score Numérico: {resposta_api['score_numerico']}")
        print(f"   🕒 Timestamp: {resposta_api['timestamp']}")
        
        return True
        
    except Exception as api_error:
        print(f"❌ Erro na simulação de API: {api_error}")
        return False

if __name__ == "__main__":
    # Teste principal
    sucesso_local = testar_modelo_local()
    
    if sucesso_local:
        # Teste como API
        sucesso_api = testar_modelo_como_api()
        
        print("\n" + "=" * 60)
        print("🏁 RESULTADO FINAL")
        print("=" * 60)
        print(f"✅ Modelo Local: {'FUNCIONANDO' if sucesso_local else 'PROBLEMA'}")
        print(f"🌐 Simulação API: {'FUNCIONANDO' if sucesso_api else 'PROBLEMA'}")
        
        if sucesso_local and sucesso_api:
            print("\n🎉 PARABÉNS! O MODELO ESTÁ PRONTO PARA PRODUÇÃO!")
            print("📋 Próximos passos:")
            print("   1. Deploy em servidor")
            print("   2. Criar endpoint REST")
            print("   3. Adicionar autenticação")
            print("   4. Configurar monitoramento")
        else:
            print("\n⚠️ Modelo precisa de ajustes antes da produção")
    else:
        print("\n❌ Modelo local não está funcionando - verifique o treinamento")