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


def main():
    """Função principal para treinamento APENAS do Random Forest com MLflow"""
    import dagshub
    
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
    
    # Treinando APENAS Random Forest COM MLflow
    try:
        print("\n🚀 Treinando Random Forest...")
        model, metrics = train_random_forest(X_train, y_train, X_val, y_val, preprocessor)
        
        # Salvar modelo localmente também
        import joblib
        import os
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_credit_score.pkl'
        joblib.dump(model, model_path)
        
        if le:
            encoder_path = 'models/label_encoder.pkl'
            joblib.dump(le, encoder_path)
            print(f"✅ Label encoder salvo: {encoder_path}")
        
        print(f"✅ Modelo salvo localmente: {model_path}")
        
        # Resumo dos resultados
        print("\n=== RESULTADOS DO RANDOM FOREST ===")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        print("\n✅ Treinamento concluído! Modelo registrado no MLflow.")
        
    except Exception as e:
        print(f"Erro no treinamento do Random Forest: {e}")
        raise


if __name__ == "__main__":
    main()