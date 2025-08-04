#!/usr/bin/env python3
"""
Código para registrar modelo no DagsHub usando método compatível
Carrega modelo existente e re-registra com registered_model_name
"""

import mlflow
import mlflow.pyfunc
import dagshub
import joblib
import os

def registrar_modelo_dagshub():
    """Registra modelo no DagsHub usando método compatível"""
    
    print("=" * 60)
    print("🔗 REGISTRO DE MODELO NO DAGSHUB")
    print("=" * 60)
    print("🎯 Usando método compatível com DagsHub")
    print()
    
    # Configurar MLflow
    print("🔧 Configurando MLflow para DagsHub...")
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    tracking_uri = "https://dagshub.com/domires/fiap-mlops-score-model.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    
    print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
    print()
    
    # Verificar se modelo local existe
    if not os.path.exists('models/random_forest_credit_score.pkl'):
        print("❌ Modelo local não encontrado!")
        print("💡 Execute primeiro: python train_credit_score_model.py")
        return
    
    print("📊 Carregando modelo local...")
    
    try:
        # Carregar modelo e encoder salvos localmente
        modelo = joblib.load('models/random_forest_credit_score.pkl')
        label_encoder = None
        
        if os.path.exists('models/label_encoder.pkl'):
            label_encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Label encoder carregado")
        
        print("✅ Modelo carregado com sucesso!")
        
        # Criar wrapper personalizado
        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model, label_encoder=None):
                self.model = model
                self.label_encoder = label_encoder
            
            def predict(self, context, model_input):
                predictions = self.model.predict(model_input)
                if self.label_encoder:
                    try:
                        return self.label_encoder.inverse_transform(predictions)
                    except:
                        return predictions
                return predictions
        
        wrapped_model = ModelWrapper(modelo, label_encoder)
        
        # Iniciar novo run para registro
        with mlflow.start_run(run_name="Registro Manual - fiap-mlops-score-model"):
            print("\n🚀 Registrando modelo usando método compatível...")
            
            # MÉTODO COMPATÍVEL: log_model com registered_model_name
            model_info = mlflow.pyfunc.log_model(
                artifact_path="credit_score_model",
                python_model=wrapped_model,
                registered_model_name="fiap-mlops-score-model",  # ← Registro automático
                pip_requirements=["scikit-learn", "pandas", "numpy"]
            )
            
            print("✅ SUCESSO! Modelo registrado!")
            print(f"🔗 Nome: fiap-mlops-score-model")
            print(f"📊 Artifact path: credit_score_model")
            print(f"📊 Run ID: {mlflow.active_run().info.run_id}")
            print()
            print("🎯 VERIFIQUE:")
            print("   1. Aba 'Models' no MLflow UI")
            print("   2. Modelo 'fiap-mlops-score-model' registrado")
            print("   3. Nova versão disponível")
            print()
            print("✅ MÉTODO FUNCIONOU! DagsHub aceita este formato")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        print("💡 Verifique se os arquivos do modelo existem")

if __name__ == "__main__":
    registrar_modelo_dagshub()