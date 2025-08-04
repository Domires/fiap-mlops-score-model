#!/usr/bin/env python3
"""
API REAL PARA CLASSIFICAÇÃO DE CREDIT SCORE
Endpoint pronto para uso em produção
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS para chamadas de outros domínios

# Carregar modelo uma vez na inicialização
print("🚀 INICIANDO API DE CREDIT SCORE...")
print("📦 Carregando modelo...")

try:
    modelo = joblib.load('models/random_forest_credit_score.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    
    with open('models/api_info.json', 'r') as f:
        api_info = json.load(f)
    
    print(f"✅ Modelo carregado: {api_info['model_name']} v{api_info['version']}")
    print(f"📊 Classes: {api_info['classes']}")
    print(f"📊 Features esperadas: {api_info['features_count']}")
    
except Exception as e:
    print(f"❌ ERRO ao carregar modelo: {e}")
    exit(1)

@app.route('/', methods=['GET'])
def home():
    """Endpoint de informações da API"""
    return jsonify({
        "api_name": "Credit Score Classification API",
        "version": api_info['version'],
        "model": api_info['model_name'],
        "status": "online",
        "endpoints": {
            "POST /classify": "Classificar score de crédito",
            "GET /info": "Informações detalhadas da API",
            "GET /health": "Health check",
            "POST /classify/batch": "Classificação em lote"
        },
        "example_usage": {
            "endpoint": "POST /classify",
            "content_type": "application/json",
            "body": "Veja /info para exemplo completo"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "api_version": api_info['version']
    })

@app.route('/info', methods=['GET'])
def info():
    """Informações detalhadas da API e exemplo de uso"""
    
    exemplo_body = {
        "cliente_id": "CLT_12345",
        "Month": "January",
        "Age": 35,
        "Occupation": "Engineer", 
        "Annual_Income": 75000,
        "Monthly_Inhand_Salary": 6250,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 2,
        "Interest_Rate": 8,
        "Num_of_Loan": 1,
        "Type_of_Loan": "Home Loan",
        "Delay_from_due_date": 0,
        "Num_of_Delayed_Payment": 0,
        "Changed_Credit_Limit": 5.5,
        "Num_Credit_Inquiries": 2,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 2500,
        "Credit_Utilization_Ratio": 25.8,
        "Credit_History_Age": "10 Years and 6 Months",
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 450,
        "Amount_invested_monthly": 200,
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Monthly_Balance": 400
    }
    
    return jsonify({
        "api_info": {
            "name": api_info['model_name'],
            "version": api_info['version'],
            "framework": api_info['framework'],
            "model_type": api_info['model_type'],
            "classes": api_info['classes'],
            "metrics": api_info['metrics']
        },
        "usage": {
            "endpoint": "POST http://localhost:5000/classify",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body_example": exemplo_body,
            "required_fields": api_info['expected_columns']
        },
        "response_format": {
            "credit_score": "Good | Standard",
            "confidence": "70.5%",
            "probabilities": {
                "Good": "29.5%",
                "Standard": "70.5%"
            },
            "cliente_id": "CLT_12345",
            "timestamp": "2025-08-03T22:30:00",
            "processing_time_ms": 2.5,
            "status": "success"
        }
    })

@app.route('/classify', methods=['POST'])
def classify_credit_score():
    """Endpoint principal para classificação de score de crédito"""
    
    try:
        start_time = datetime.now()
        
        # Validar JSON
        if not request.is_json:
            return jsonify({
                "error": "Content-Type deve ser application/json",
                "status": "error"
            }), 400
        
        dados = request.get_json()
        
        # Validar campos obrigatórios
        campos_necessarios = api_info['expected_columns']
        campos_ausentes = [campo for campo in campos_necessarios 
                          if campo not in dados and campo != 'cliente_id']
        
        if campos_ausentes:
            return jsonify({
                "error": f"Campos obrigatórios ausentes: {campos_ausentes}",
                "required_fields": campos_necessarios,
                "status": "error"
            }), 400
        
        # Extrair cliente_id se presente
        cliente_id = dados.get('cliente_id', f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Preparar dados para o modelo (remover cliente_id)
        dados_modelo = {k: v for k, v in dados.items() if k != 'cliente_id'}
        df_cliente = pd.DataFrame([dados_modelo])
        
        # Fazer predição
        predicao = modelo.predict(df_cliente)
        score = encoder.inverse_transform(predicao)[0]
        
        # Obter probabilidades
        probabilities = {}
        confidence = None
        
        if hasattr(modelo, 'predict_proba'):
            probas = modelo.predict_proba(df_cliente)[0]
            classes = encoder.classes_
            
            for i, classe in enumerate(classes):
                probabilities[classe] = f"{probas[i]:.1%}"
            
            confidence = f"{max(probas):.1%}"
        
        # Calcular tempo de processamento
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Resposta estruturada
        resposta = {
            "credit_score": score,
            "confidence": confidence,
            "probabilities": probabilities,
            "cliente_id": cliente_id,
            "timestamp": end_time.isoformat(),
            "processing_time_ms": round(processing_time, 2),
            "status": "success"
        }
        
        logger.info(f"Classificação realizada - Cliente: {cliente_id}, Score: {score}, Tempo: {processing_time:.2f}ms")
        
        return jsonify(resposta)
        
    except Exception as e:
        logger.error(f"Erro na classificação: {str(e)}")
        return jsonify({
            "error": f"Erro interno: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """Endpoint para classificação em lote"""
    
    try:
        start_time = datetime.now()
        
        if not request.is_json:
            return jsonify({
                "error": "Content-Type deve ser application/json",
                "status": "error"
            }), 400
        
        dados = request.get_json()
        
        # Validar se é uma lista
        if not isinstance(dados, list):
            return jsonify({
                "error": "Body deve ser uma lista de clientes",
                "example": [{"cliente_id": "CLT_001", "Age": 35, "...": "outros campos"}],
                "status": "error"
            }), 400
        
        resultados = []
        
        for i, cliente in enumerate(dados):
            try:
                # Usar endpoint individual para cada cliente
                with app.test_request_context('/classify', json=cliente, method='POST'):
                    resultado = classify_credit_score()
                    if resultado[1] == 200:  # Status code 200
                        resultados.append(resultado[0].get_json())
                    else:
                        resultados.append({
                            "error": f"Erro no cliente {i}",
                            "details": resultado[0].get_json(),
                            "index": i
                        })
            except Exception as e:
                resultados.append({
                    "error": f"Erro no cliente {i}: {str(e)}",
                    "index": i
                })
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return jsonify({
            "total_clientes": len(dados),
            "processados_com_sucesso": len([r for r in resultados if "error" not in r]),
            "resultados": resultados,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": end_time.isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Erro no processamento em lote: {str(e)}",
            "status": "error"
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 API CREDIT SCORE RODANDO!")
    print("="*60)
    print("📍 URL Base: http://localhost:5000")
    print("🎯 Endpoint Principal: POST /classify")
    print("📊 Informações: GET /info")
    print("💚 Health Check: GET /health")
    print("📦 Lote: POST /classify/batch")
    print("="*60)
    print()
    
    # Rodar servidor
    app.run(
        host='0.0.0.0',  # Acessível de qualquer IP
        port=5000,
        debug=True
    )