#!/usr/bin/env python3
"""
EXEMPLO COMPLETO DE USO DA API
Como fazer requisições para classificar score de crédito
"""

import requests
import json
from datetime import datetime

# URL da API (ajuste se necessário)
BASE_URL = "http://localhost:5000"

def testar_api_credit_score():
    """Testa todos os endpoints da API"""
    
    print("=" * 80)
    print("🧪 TESTANDO API DE CREDIT SCORE")
    print("=" * 80)
    print(f"🌐 URL Base: {BASE_URL}")
    print()
    
    # ===== TESTE 1: Health Check =====
    print("1️⃣ TESTE HEALTH CHECK")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API está online!")
            print(f"📊 Resposta: {response.json()}")
        else:
            print(f"❌ API offline - Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Não foi possível conectar na API")
        print("💡 Certifique-se que está rodando: python api_credit_score.py")
        return False
    
    # ===== TESTE 2: Informações da API =====
    print(f"\n2️⃣ INFORMAÇÕES DA API")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/info")
    if response.status_code == 200:
        info = response.json()
        print(f"✅ API: {info['api_info']['name']}")
        print(f"📊 Versão: {info['api_info']['version']}")
        print(f"🎯 Classes: {info['api_info']['classes']}")
        print(f"📋 Endpoint: {info['usage']['endpoint']}")
    
    # ===== TESTE 3: Classificação Individual =====
    print(f"\n3️⃣ TESTE CLASSIFICAÇÃO INDIVIDUAL")
    print("-" * 40)
    
    # Cliente de exemplo
    cliente_exemplo = {
        "cliente_id": "CLT_TESTE_001",
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
    
    print("📤 Enviando dados do cliente...")
    print(f"👤 Cliente ID: {cliente_exemplo['cliente_id']}")
    print(f"💰 Renda: ${cliente_exemplo['Annual_Income']:,}")
    print(f"👥 Ocupação: {cliente_exemplo['Occupation']}")
    
    response = requests.post(
        f"{BASE_URL}/classify",
        json=cliente_exemplo,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        resultado = response.json()
        print(f"\n📊 RESULTADO DA CLASSIFICAÇÃO:")
        print(f"   🎯 Credit Score: {resultado['credit_score']}")
        print(f"   📊 Confiança: {resultado['confidence']}")
        print(f"   ⏱️ Tempo processamento: {resultado['processing_time_ms']}ms")
        print(f"   📅 Timestamp: {resultado['timestamp']}")
        
        if 'probabilities' in resultado:
            print(f"   📈 Probabilidades:")
            for classe, prob in resultado['probabilities'].items():
                print(f"      - {classe}: {prob}")
        
        print(f"   ✅ Status: {resultado['status']}")
        
    else:
        print(f"❌ Erro na classificação: {response.status_code}")
        print(f"📋 Resposta: {response.json()}")
    
    # ===== TESTE 4: Diferentes Perfis =====
    print(f"\n4️⃣ TESTE COM DIFERENTES PERFIS")
    print("-" * 40)
    
    perfis = [
        {
            "nome": "Cliente Bom",
            "dados": {
                "cliente_id": "CLT_BOM_001",
                "Month": "June",
                "Age": 45,
                "Occupation": "Doctor",
                "Annual_Income": 120000,
                "Monthly_Inhand_Salary": 10000,
                "Num_Bank_Accounts": 4,
                "Num_Credit_Card": 2,
                "Interest_Rate": 6,
                "Num_of_Loan": 1,
                "Type_of_Loan": "Home Loan",
                "Delay_from_due_date": 0,
                "Num_of_Delayed_Payment": 0,
                "Changed_Credit_Limit": 8.0,
                "Num_Credit_Inquiries": 1,
                "Credit_Mix": "Good",
                "Outstanding_Debt": 1500,
                "Credit_Utilization_Ratio": 15.5,
                "Credit_History_Age": "20 Years and 8 Months",
                "Payment_of_Min_Amount": "Yes",
                "Total_EMI_per_month": 300,
                "Amount_invested_monthly": 500,
                "Payment_Behaviour": "High_spent_Small_value_payments",
                "Monthly_Balance": 800
            }
        },
        {
            "nome": "Cliente Arriscado",
            "dados": {
                "cliente_id": "CLT_RISCO_001",
                "Month": "March",
                "Age": 25,
                "Occupation": "Student",
                "Annual_Income": 25000,
                "Monthly_Inhand_Salary": 2000,
                "Num_Bank_Accounts": 1,
                "Num_Credit_Card": 5,
                "Interest_Rate": 22,
                "Num_of_Loan": 3,
                "Type_of_Loan": "Personal Loan",
                "Delay_from_due_date": 15,
                "Num_of_Delayed_Payment": 8,
                "Changed_Credit_Limit": -2.5,
                "Num_Credit_Inquiries": 6,
                "Credit_Mix": "Bad",
                "Outstanding_Debt": 7500,
                "Credit_Utilization_Ratio": 85.2,
                "Credit_History_Age": "2 Years and 3 Months",
                "Payment_of_Min_Amount": "No",
                "Total_EMI_per_month": 750,
                "Amount_invested_monthly": 0,
                "Payment_Behaviour": "Low_spent_Large_value_payments",
                "Monthly_Balance": 50
            }
        }
    ]
    
    for perfil in perfis:
        print(f"\n👤 Testando: {perfil['nome']}")
        print(f"   💰 Renda: ${perfil['dados']['Annual_Income']:,}")
        print(f"   📊 Atrasos: {perfil['dados']['Num_of_Delayed_Payment']}")
        
        response = requests.post(
            f"{BASE_URL}/classify",
            json=perfil['dados'],
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            resultado = response.json()
            print(f"   🎯 Score: {resultado['credit_score']}")
            print(f"   📊 Confiança: {resultado['confidence']}")
        else:
            print(f"   ❌ Erro: {response.status_code}")
    
    # ===== TESTE 5: Classificação em Lote =====
    print(f"\n5️⃣ TESTE CLASSIFICAÇÃO EM LOTE")
    print("-" * 40)
    
    lote_clientes = [perfil['dados'] for perfil in perfis]
    
    response = requests.post(
        f"{BASE_URL}/classify/batch",
        json=lote_clientes,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        resultado = response.json()
        print(f"📊 Total clientes: {resultado['total_clientes']}")
        print(f"✅ Processados com sucesso: {resultado['processados_com_sucesso']}")
        print(f"⏱️ Tempo total: {resultado['processing_time_ms']}ms")
        
        print(f"\n📋 Resultados individuais:")
        for i, res in enumerate(resultado['resultados']):
            if 'error' not in res:
                print(f"   {i+1}. {res['cliente_id']}: {res['credit_score']} ({res['confidence']})")
            else:
                print(f"   {i+1}. Erro: {res['error']}")
    else:
        print(f"❌ Erro no lote: {response.status_code}")
    
    print(f"\n✅ TESTES CONCLUÍDOS!")
    return True

def exemplo_codigo_uso():
    """Mostra exemplos de código para usar a API"""
    
    print(f"\n" + "=" * 80)
    print("📝 EXEMPLOS DE CÓDIGO PARA USAR A API")
    print("=" * 80)
    
    print("""
🐍 PYTHON:
---------
import requests

# 1. Classificação individual
dados_cliente = {
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

response = requests.post(
    "http://localhost:5000/classify",
    json=dados_cliente,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    resultado = response.json()
    print(f"Credit Score: {resultado['credit_score']}")
    print(f"Confiança: {resultado['confidence']}")
else:
    print(f"Erro: {response.status_code}")
""")
    
    print("""
🌐 cURL:
-------
curl -X POST http://localhost:5000/classify \\
  -H "Content-Type: application/json" \\
  -d '{
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
  }'
""")
    
    print("""
💻 JavaScript/Node.js:
--------------------
const axios = require('axios');

const dadosCliente = {
  cliente_id: "CLT_12345",
  Month: "January",
  Age: 35,
  Occupation: "Engineer",
  Annual_Income: 75000,
  Monthly_Inhand_Salary: 6250,
  Num_Bank_Accounts: 3,
  Num_Credit_Card: 2,
  Interest_Rate: 8,
  Num_of_Loan: 1,
  Type_of_Loan: "Home Loan",
  Delay_from_due_date: 0,
  Num_of_Delayed_Payment: 0,
  Changed_Credit_Limit: 5.5,
  Num_Credit_Inquiries: 2,
  Credit_Mix: "Good",
  Outstanding_Debt: 2500,
  Credit_Utilization_Ratio: 25.8,
  Credit_History_Age: "10 Years and 6 Months",
  Payment_of_Min_Amount: "Yes",
  Total_EMI_per_month: 450,
  Amount_invested_monthly: 200,
  Payment_Behaviour: "High_spent_Small_value_payments",
  Monthly_Balance: 400
};

axios.post('http://localhost:5000/classify', dadosCliente)
  .then(response => {
    console.log('Credit Score:', response.data.credit_score);
    console.log('Confiança:', response.data.confidence);
  })
  .catch(error => {
    console.error('Erro:', error.response.data);
  });
""")

if __name__ == "__main__":
    print("🚀 INICIANDO TESTES DA API...")
    
    sucesso = testar_api_credit_score()
    
    if sucesso:
        exemplo_codigo_uso()
        
        print(f"\n🎉 TESTES CONCLUÍDOS COM SUCESSO!")
        print(f"📋 A API está funcionando perfeitamente!")
    else:
        print(f"\n❌ Problemas encontrados nos testes")
        print(f"💡 Certifique-se que a API está rodando: python api_credit_score.py")