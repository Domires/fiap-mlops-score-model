"""
Módulo para treinamento de modelos de classificação de credit score.
"""



import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from mlflow.models import infer_signature
from data_preprocessing import load_and_preprocess_data, create_preprocessing_pipeline


def evaluate_and_log_classification_model(kind, model_name, pipeline, X_val, y_val):
    """
    Função para avaliar e registrar modelos de classificação
    
    Args:
        kind (str): Tipo do modelo (sklearn, xgboost, lightgbm, catboost)
        model_name (str): Nome do modelo
        pipeline: Pipeline treinado
        X_val: Features de validação
        y_val: Target de validação
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    # Fazendo predições
    predictions = pipeline.predict(X_val)
    prediction_proba = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
    
    # Calculando métricas
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average='weighted')
    recall = recall_score(y_val, predictions, average='weighted')
    f1 = f1_score(y_val, predictions, average='weighted')
    
    # AUC-ROC para classificação multiclasse
    if prediction_proba is not None:
        try:
            auc_roc = roc_auc_score(y_val, prediction_proba, multi_class='ovr', average='weighted')
        except:
            auc_roc = 0.0
    else:
        auc_roc = 0.0
    
    # Registrando métricas no MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    
    # Criando assinatura do modelo
    signature = infer_signature(X_val, predictions)
    
    # Registrando modelo baseado no tipo
    if kind == "catboost":
        mlflow.catboost.log_model(pipeline.named_steps['classifier'], model_name, signature=signature, input_example=X_val[:5])
    elif kind == "xgboost":
        mlflow.xgboost.log_model(pipeline.named_steps['classifier'], model_name, signature=signature, input_example=X_val[:5])
    elif kind == "lightgbm":
        mlflow.lightgbm.log_model(pipeline.named_steps['classifier'], model_name, signature=signature, input_example=X_val[:5])
    else:
        mlflow.sklearn.log_model(pipeline, model_name, signature=signature, input_example=X_val[:5])
    
    print(f"Modelo {model_name} avaliado:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, preprocessor):
    """Treina modelo de Regressão Logística"""
    with mlflow.start_run(run_name="Logistic Regression - Credit Score"):
        # Criando pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Definindo parâmetros para busca
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Melhor modelo
        best_pipeline = grid_search.best_estimator_
        
        # Registrando melhores parâmetros
        mlflow.log_param("best_C", grid_search.best_params_['classifier__C'])
        mlflow.log_param("best_solver", grid_search.best_params_['classifier__solver'])
        
        # Avaliando modelo
        metrics = evaluate_and_log_classification_model("sklearn", "logistic_regression", best_pipeline, X_val, y_val)
        
        return best_pipeline, metrics


def train_random_forest(X_train, y_train, X_val, y_val, preprocessor):
    """Treina modelo Random Forest"""
    with mlflow.start_run(run_name="Random Forest - Credit Score"):
        # Criando pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Definindo parâmetros para busca
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Melhor modelo
        best_pipeline = grid_search.best_estimator_
        
        # Registrando melhores parâmetros
        mlflow.log_param("best_n_estimators", grid_search.best_params_['classifier__n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['classifier__max_depth'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['classifier__min_samples_split'])
        
        # Avaliando modelo
        metrics = evaluate_and_log_classification_model("sklearn", "random_forest", best_pipeline, X_val, y_val)
        
        return best_pipeline, metrics


def train_xgboost(X_train, y_train, X_val, y_val, preprocessor):
    """Treina modelo XGBoost"""
    with mlflow.start_run(run_name="XGBoost - Credit Score"):
        # Criando pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        
        # Definindo parâmetros para busca
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.1, 0.2]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Melhor modelo
        best_pipeline = grid_search.best_estimator_
        
        # Registrando melhores parâmetros
        mlflow.log_param("best_n_estimators", grid_search.best_params_['classifier__n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['classifier__max_depth'])
        mlflow.log_param("best_learning_rate", grid_search.best_params_['classifier__learning_rate'])
        
        # Avaliando modelo
        metrics = evaluate_and_log_classification_model("xgboost", "xgboost_classifier", best_pipeline, X_val, y_val)
        
        return best_pipeline, metrics


def train_lightgbm(X_train, y_train, X_val, y_val, preprocessor):
    """Treina modelo LightGBM"""
    with mlflow.start_run(run_name="LightGBM - Credit Score"):
        # Criando pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])
        
        # Definindo parâmetros para busca
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.1, 0.2],
            'classifier__num_leaves': [31, 50]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Melhor modelo
        best_pipeline = grid_search.best_estimator_
        
        # Registrando melhores parâmetros
        mlflow.log_param("best_n_estimators", grid_search.best_params_['classifier__n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['classifier__max_depth'])
        mlflow.log_param("best_learning_rate", grid_search.best_params_['classifier__learning_rate'])
        mlflow.log_param("best_num_leaves", grid_search.best_params_['classifier__num_leaves'])
        
        # Avaliando modelo
        metrics = evaluate_and_log_classification_model("lightgbm", "lightgbm_classifier", best_pipeline, X_val, y_val)
        
        return best_pipeline, metrics


def main_random_forest_mlflow():
    """Função para treinamento do Random Forest COM MLflow"""
    import dagshub
    
    print("🎯 TREINAMENTO ÚNICO - RANDOM FOREST COM MLFLOW")
    print("✅ APENAS 1 modelo será registrado no MLflow")
    print("✅ SEM múltiplos modelos")
    
    # Configuração do MLflow (sem autolog para controle manual)
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/domires/fiap-mlops-score-model.mlflow")
    
    try:
        # Carregando dados usando caminhos alternativos
        try:
            train_processed, test_processed, features = load_and_preprocess_data(
                'references/exemplo_train.csv',
                'references/exemplo_test.csv'
            )
        except:
            # Fallback para outros caminhos possíveis
            train_processed, test_processed, features = load_and_preprocess_data(
                'data/raw/train.csv',
                'data/raw/test.csv'
            )
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        print("💡 Verifique se os arquivos de dados estão disponíveis")
        return
    
    # Separando features e target
    X = train_processed[features]
    y = train_processed['Credit_Score']
    
    # Tratamento para target string (conversão automática)
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"🔄 Target convertido: {le.classes_} → {range(len(le.classes_))}")
    else:
        y_encoded = y
        le = None
    
    # Criando pipeline de pré-processamento
    preprocessor = create_preprocessing_pipeline(X)
    
    # Dividindo dados para treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    print(f"📊 Dados preparados:")
    print(f"   Treino: {X_train.shape}")
    print(f"   Validação: {X_val.shape}")
    
    # Treinando APENAS Random Forest COM MLflow
    with mlflow.start_run(run_name="Random Forest - Credit Score (Único Modelo)"):
        print("\n🚀 Treinando Random Forest...")
        
        from sklearn.pipeline import Pipeline
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        # Registrando parâmetros no MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 30)
        mlflow.log_param("min_samples_split", 2)
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("max_features", "sqrt")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)
        
        # Treinar modelo
        rf_pipeline.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = rf_pipeline.predict(X_val)
        y_pred_proba = rf_pipeline.predict_proba(X_val)
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Registrar métricas no MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Registrar modelo no MLflow
        from mlflow.models import infer_signature
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(rf_pipeline, "random_forest_model", signature=signature)
        
        print(f"\n📊 RESULTADOS DO RANDOM FOREST:")
        print(f"   🎯 Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 Precisão:  {precision:.4f}")
        print(f"   🎯 Recall:    {recall:.4f}")
        print(f"   🎯 F1-Score:  {f1:.4f}")
        
        # Salvar modelo localmente também
        import joblib
        import os
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_credit_score.pkl'
        joblib.dump(rf_pipeline, model_path)
        
        if le:
            encoder_path = 'models/label_encoder.pkl'
            joblib.dump(le, encoder_path)
            print(f"✅ Label encoder salvo: {encoder_path}")
        
        print(f"✅ Modelo salvo localmente: {model_path}")
        print(f"✅ Modelo registrado no MLflow!")
        print(f"✅ APENAS 1 modelo Random Forest foi treinado!")


def main_random_forest_only():
    """Função para treinamento APENAS do Random Forest (SEM MLflow)"""
    import joblib
    import os
    
    print("🎯 TREINAMENTO ÚNICO - RANDOM FOREST")
    print("✅ SEM MLflow (evita problemas de endpoint)")
    print("✅ SEM múltiplos modelos")
    
    try:
        # Carregando dados usando caminhos alternativos
        try:
            train_processed, test_processed, features = load_and_preprocess_data(
                'references/exemplo_train.csv',
                'references/exemplo_test.csv'
            )
        except:
            # Fallback para outros caminhos possíveis
            train_processed, test_processed, features = load_and_preprocess_data(
                'data/raw/train.csv',
                'data/raw/test.csv'
            )
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        print("💡 Verifique se os arquivos de dados estão disponíveis")
        return
    
    # Separando features e target
    X = train_processed[features]
    y = train_processed['Credit_Score']
    
    # Tratamento para target string (conversão automática)
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"🔄 Target convertido: {le.classes_} → {range(len(le.classes_))}")
    else:
        y_encoded = y
        le = None
    
    # Criando pipeline de pré-processamento
    preprocessor = create_preprocessing_pipeline(X)
    
    # Dividindo dados para treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    print(f"📊 Dados preparados:")
    print(f"   Treino: {X_train.shape}")
    print(f"   Validação: {X_val.shape}")
    
    # Treinando APENAS Random Forest (sem MLflow)
    print("\n🚀 Treinando Random Forest...")
    
    from sklearn.pipeline import Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Treinar modelo
    rf_pipeline.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = rf_pipeline.predict(X_val)
    
    # Calcular métricas
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"\n📊 RESULTADOS DO RANDOM FOREST:")
    print(f"   🎯 Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   🎯 Precisão:  {precision:.4f}")
    print(f"   🎯 Recall:    {recall:.4f}")
    print(f"   🎯 F1-Score:  {f1:.4f}")
    
    # Salvar modelo
    os.makedirs('models', exist_ok=True)
    model_path = 'models/random_forest_credit_score.pkl'
    joblib.dump(rf_pipeline, model_path)
    
    if le:
        encoder_path = 'models/label_encoder.pkl'
        joblib.dump(le, encoder_path)
        print(f"✅ Label encoder salvo: {encoder_path}")
    
    print(f"✅ Modelo salvo: {model_path}")
    print(f"✅ APENAS 1 modelo Random Forest foi treinado!")


def register_existing_model(run_id, model_name="credit_score_random_forest"):
    """
    Registra um modelo existente usando run_id específico conforme documentação do curso
    
    Args:
        run_id (str): ID do run que contém o modelo (ex: "054a9cedbf3341f1910b8ff2ee49490a")
        model_name (str): Nome para registrar o modelo
    """
    import dagshub
    import mlflow
    
    print(f"🔗 === REGISTRANDO MODELO EXISTENTE ===")
    print(f"📊 Run ID fornecido: {run_id}")
    print(f"🎯 Nome do modelo: {model_name}")
    
    # Configurar MLflow
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    tracking_uri = "https://dagshub.com/domires/fiap-mlops-score-model.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    
    try:
        # Montar model_uri usando o run_id específico
        model_uri = f"runs:/{run_id}/random_forest_model"
        print(f"🔗 Model URI: {model_uri}")
        
        # Registrar modelo conforme documentação do curso
        print("🚀 Registrando modelo usando mlflow.register_model()...")
        registered_model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print("✅ SUCESSO! Modelo registrado!")
        print(f"🔗 Modelo: {model_name}")
        print(f"📊 Versão: {registered_model_version.version}")
        print(f"📊 Run ID: {run_id}")
        print("🎯 Verifique na aba 'Models' do MLflow UI")
        print("📋 Conforme documentação: A cada novo registro uma nova versão será gerada")
        
        return registered_model_version
        
    except Exception as e:
        print(f"❌ Erro ao registrar modelo: {e}")
        print("💡 Verifique se o run_id existe e contém o modelo")
        return None


def main():
    """Função principal para treinamento APENAS do Random Forest com MLflow simplificado"""
    import dagshub
    import mlflow
    import mlflow.pyfunc
    
    # Configuração completa do MLflow para Model Registry
    print("🔧 Configurando MLflow + DagsHub...")
    
    # Inicializar DagsHub
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    
    # Configurar URIs para tracking E registry
    tracking_uri = "https://dagshub.com/domires/fiap-mlops-score-model.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)  # ← CRUCIAL para Model Registry
    
    print(f"✅ Tracking URI: {tracking_uri}")
    print(f"✅ Registry URI: {mlflow.get_registry_uri()}")
    print(f"✅ Current Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Verificar conectividade
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ Conectado! {len(experiments)} experimentos encontrados")
    except Exception as conn_error:
        print(f"⚠️ Problema de conectividade: {conn_error}")
        print("🔧 Continuando mesmo assim...")
    
    try:
        # Carregando dados usando caminhos alternativos
        try:
            train_processed, test_processed, features = load_and_preprocess_data(
                'references/exemplo_train.csv',
                'references/exemplo_test.csv'
            )
        except:
            # Fallback para outros caminhos possíveis
            train_processed, test_processed, features = load_and_preprocess_data(
                'data/raw/train.csv',
                'data/raw/test.csv'
            )
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        print("💡 Verifique se os arquivos de dados estão disponíveis")
        return
    
    # Separando features e target
    X = train_processed[features]
    y = train_processed['Credit_Score']
    
    # Tratamento para target com valores NaN
    if y.isna().sum() > 0:
        print(f"🔄 Convertendo {y.isna().sum()} valores NaN no target para 'Unknown'")
        y = y.fillna('Unknown')
    
    # Tratamento para target string (conversão automática)
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"🔄 Target convertido: {le.classes_} → {range(len(le.classes_))}")
    else:
        y_encoded = y
        le = None
    
    # Criando pipeline de pré-processamento
    preprocessor = create_preprocessing_pipeline(X)
    
    # Dividindo dados para treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    print(f"📊 Dados preparados:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # TREINAMENTO SIMPLIFICADO SEM PROBLEMAS DE ENDPOINT
    with mlflow.start_run(run_name="Random Forest - Credit Score (Único Modelo)"):
        print("\n🚀 Treinando Random Forest...")
        
        # Criar o pipeline completo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        # Parâmetros ótimos baseados na referência
        rf_params = {
            'n_estimators': 150,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log dos parâmetros
        for param, value in rf_params.items():
            mlflow.log_param(f"rf_{param}", value)
        
        # Criar pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**rf_params))
        ])
        
        # Treinar o modelo
        pipeline.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = pipeline.predict(X_val)
        y_pred_proba = pipeline.predict_proba(X_val)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # AUC-ROC (multiclass)
        if len(set(y_val)) > 2:
            auc_roc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            auc_roc = roc_auc_score(y_val, y_pred_proba[:, 1])
        
        # Log das métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)
        
        # Salvar modelo localmente E registrar no MLflow
        import joblib
        import os
        import tempfile
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_credit_score.pkl'
        joblib.dump(pipeline, model_path)
        
        if le:
            encoder_path = 'models/label_encoder.pkl'
            joblib.dump(le, encoder_path)
            print(f"✅ Label encoder salvo: {encoder_path}")
        
        print(f"✅ Modelo salvo localmente: {model_path}")
        
        # Registrar modelo no Model Registry com debugging completo
        print("\n🔗 === REGISTRANDO MODELO NO MODEL REGISTRY ===")
        print(f"📊 Registry URI ativo: {mlflow.get_registry_uri()}")
        print(f"📊 Tracking URI ativo: {mlflow.get_tracking_uri()}")
        print(f"📊 Run ID atual: {mlflow.active_run().info.run_id}")
        
        # Criar wrapper customizado para o modelo (pronto para API)
        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model, label_encoder=None):
                self.model = model
                self.label_encoder = label_encoder
            
            def predict(self, context, model_input):
                """
                Predição para API de Credit Score
                
                Input: DataFrame com features do cliente
                Output: Array com predições (Good, Standard, Unknown)
                """
                # Fazer predição numérica
                predictions = self.model.predict(model_input)
                
                # Converter de volta para labels legíveis se tiver label encoder
                if self.label_encoder:
                    try:
                        # Converter predições numéricas para strings originais
                        readable_predictions = self.label_encoder.inverse_transform(predictions)
                        return readable_predictions
                    except Exception:
                        # Se falhar, retornar predições numéricas
                        return predictions
                
                return predictions
            
            def predict_proba(self, context, model_input):
                """
                Probabilidades para cada classe (se disponível)
                """
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(model_input)
                else:
                    # Se não tiver predict_proba, retornar predições binárias
                    predictions = self.predict(context, model_input)
                    # Simular probabilidades (1.0 para classe predita, 0.0 para outras)
                    import numpy as np
                    n_samples = len(predictions)
                    n_classes = len(np.unique(predictions))
                    proba = np.zeros((n_samples, n_classes))
                    for i, pred in enumerate(predictions):
                        proba[i, pred] = 1.0
                    return proba
        
        wrapped_model = ModelWrapper(pipeline, le)
        model_name = "credit_score_random_forest"
        
        try:
            # PREPARAR SIGNATURE E INPUT EXAMPLE PARA API
            print("🔧 Preparando signature e input example para API...")
            
            # Criar signature do modelo (tipos de entrada e saída)
            import mlflow.types
            from mlflow.models.signature import infer_signature
            
            # Inferir signature dos dados de treino
            y_pred_train = pipeline.predict(X_train)  # Calcular predições para signature
            signature = infer_signature(X_train, y_pred_train)
            print(f"✅ Signature criada: {len(X_train.columns)} features de entrada")
            
            # Preparar input example (amostra dos dados para documentação)
            input_example = X_train.head(3)  # 3 exemplos
            print(f"✅ Input example preparado: {input_example.shape[0]} amostras")
            
            # MÉTODO 1: Log do modelo com signature e input example
            print(f"🚀 Step 1: Fazendo log do modelo como '{model_name}' com signature...")
            
            model_info = mlflow.pyfunc.log_model(
                artifact_path="random_forest_model",
                python_model=wrapped_model,
                signature=signature,  # ← CRUCIAL para API
                input_example=input_example,  # ← Exemplo para documentação
                pip_requirements=["scikit-learn", "pandas", "numpy"]
            )
            print("✅ Modelo logado com signature e input example!")
            
            # MÉTODO OFICIAL DO CURSO: mlflow.register_model() com run_id
            current_run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{current_run_id}/random_forest_model"
            
            print(f"🚀 Step 2: Registrando modelo usando mlflow.register_model()...")
            print(f"📊 Run ID atual: {current_run_id}")
            print(f"🔗 Model URI: {model_uri}")
            
            # Registrar modelo conforme documentação do curso
            registered_model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            print("✅ SUCESSO! Modelo registrado usando mlflow.register_model()!")
            print(f"🔗 Modelo: {model_name}")
            print(f"📊 Versão: {registered_model_version.version}")
            print(f"📊 Run ID: {current_run_id}")
            print("🎯 VERIFIQUE: Aba 'Models' no MLflow UI")
            print("📋 CONFORME DOCUMENTAÇÃO: A cada novo registro uma nova versão será gerada")
            print("🔌 PRONTO PARA API: Signature e input example configurados!")
            
            # Salvar documentação da API
            print("\n📋 Salvando documentação da API...")
            api_info = {
                "model_name": model_name,
                "model_version": registered_model_version.version,
                "run_id": current_run_id,
                "model_uri": model_uri,
                "features": list(X_train.columns),
                "feature_count": len(X_train.columns),
                "classes": list(le.classes_) if le else ["0", "1", "2"],
                "input_shape": list(X_train.shape),
                "signature": str(signature),
                "api_usage": {
                    "load_model": f"mlflow.pyfunc.load_model('models:/{model_name}/{registered_model_version.version}')",
                    "predict": "model.predict(input_data)",
                    "predict_proba": "model.predict_proba(input_data) # se disponível"
                }
            }
            
            # Salvar informações da API em JSON
            import json
            import os
            os.makedirs('models', exist_ok=True)
            
            with open('models/api_info.json', 'w', encoding='utf-8') as f:
                json.dump(api_info, f, indent=2, ensure_ascii=False)
            
            # Salvar exemplo de input
            input_example.to_csv('models/input_example.csv', index=False)
            
            print("✅ Arquivos criados para API:")
            print("   📄 models/api_info.json - Info do modelo")
            print("   📄 models/input_example.csv - Exemplo de entrada")
            print(f"   📄 Features necessárias: {len(X_train.columns)}")
            print(f"   📄 Classes de saída: {list(le.classes_) if le else ['0', '1', '2']}")
            
        except Exception as register_error:
            print(f"❌ Erro no registro com mlflow.register_model(): {register_error}")
            print("🔧 Tentando método manual...")
            
            try:
                # MÉTODO 2: Log primeiro, depois registrar
                print("🚀 Tentativa 2: Log + Register separados...")
                
                # Log sem registro
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="random_forest_model",
                    python_model=wrapped_model,
                    pip_requirements=["scikit-learn", "pandas", "numpy"]
                )
                print("✅ Modelo logado como artifact")
                
                # Registrar separadamente
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/random_forest_model"
                print(f"🔗 Model URI: {model_uri}")
                
                # Usar MlflowClient para mais controle
                client = mlflow.tracking.MlflowClient()
                
                # Verificar se o modelo já existe
                try:
                    existing_model = client.get_registered_model(model_name)
                    print(f"📋 Modelo '{model_name}' já existe, adicionando nova versão...")
                except Exception:
                    print(f"📋 Criando novo modelo '{model_name}'...")
                
                # Registrar no Model Registry
                registered_model = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run_id
                )
                
                print("✅ SUCESSO! Modelo registrado manualmente!")
                print(f"🔗 Modelo: {model_name}")
                print(f"📊 Versão: {registered_model.version}")
                print("🎯 VERIFIQUE: Aba 'Models' no MLflow UI")
                
            except Exception as manual_error:
                print(f"❌ Erro no registro manual: {manual_error}")
                print("🔧 Usando fallback - apenas artifacts...")
                
                try:
                    # MÉTODO 3: Apenas artifacts (para UI manual)
                    mlflow.log_artifact(model_path, "model")
                    if le:
                        mlflow.log_artifact(encoder_path, "model")
                    print("💾 Modelo salvo como artifacts")
                    print("📋 INSTRUÇÃO: Use 'Register Model' na UI manualmente")
                    print("   1. Vá para a aba 'Experiments'")
                    print("   2. Clique no run atual")
                    print("   3. Clique na pasta 'model' em Artifacts")
                    print("   4. Clique 'Register Model'")
                    
                except Exception as artifact_error:
                    print(f"❌ Erro total: {artifact_error}")
                    print("💾 Modelo salvo apenas localmente")
        
        print("🔗 === FIM DO REGISTRO ===\n")
        
        # Resumo dos resultados
        print("\n=== RESULTADOS DO RANDOM FOREST ===")
        print(f"  accuracy: {accuracy:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall: {recall:.4f}")
        print(f"  f1_score: {f1:.4f}")
        print(f"  auc_roc: {auc_roc:.4f}")
        
        print("\n✅ Treinamento concluído!")
        print("📊 Métricas registradas no MLflow")
        print("🔗 Modelo registrado para 'Register model'")
        print("💾 Modelo salvo localmente")
        
    return pipeline


if __name__ == "__main__":
    main()