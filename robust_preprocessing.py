#!/usr/bin/env python3
"""
Pré-processamento robusto para dados com muitos valores faltantes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def robust_preprocessing(df, target_col='Credit_Score', test_size=0.2):
    """
    Pré-processamento robusto para dados com muitos valores faltantes
    """
    print("=== INICIANDO PRÉ-PROCESSAMENTO ROBUSTO ===")
    print(f"Dataset original: {df.shape}")
    
    # 1. Analisar valores nulos
    null_percent = (df.isnull().sum() / len(df)) * 100
    print(f"\n📊 Colunas com mais de 70% de valores nulos:")
    high_null_cols = null_percent[null_percent > 70].index.tolist()
    for col in high_null_cols:
        print(f"  - {col}: {null_percent[col]:.1f}%")
    
    # 2. Remover colunas com muitos nulos e IDs
    cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN'] + high_null_cols
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"\n🗑️ Removidas {len([col for col in cols_to_drop if col in df.columns])} colunas")
    print(f"Dataset limpo: {df_clean.shape}")
    
    # 3. Separar features numéricas e categóricas
    if target_col in df_clean.columns:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
    else:
        X = df_clean
        y = None
    
    # Identificar tipos
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📋 Features numéricas: {len(numeric_cols)}")
    print(f"📋 Features categóricas: {len(categorical_cols)}")
    
    # 4. Pré-processamento robusto
    X_processed = X.copy()
    
    # 4a. Features numéricas - imputar com mediana
    if numeric_cols:
        for col in numeric_cols:
            median_val = X_processed[col].median()
            if pd.isna(median_val):  # Se tudo for NaN, usar 0
                median_val = 0
            X_processed[col] = X_processed[col].fillna(median_val)
            print(f"  ✅ {col}: imputado com {median_val}")
    
    # 4b. Features categóricas - imputar com moda ou 'Unknown'
    if categorical_cols:
        for col in categorical_cols:
            mode_val = X_processed[col].mode()
            if len(mode_val) > 0:
                fill_val = mode_val[0]
            else:
                fill_val = 'Unknown'
            X_processed[col] = X_processed[col].fillna(fill_val)
            print(f"  ✅ {col}: imputado com '{fill_val}'")
    
    # 5. Codificar variáveis categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
        print(f"  🔢 {col}: codificado ({len(le.classes_)} classes)")
    
    # 6. Processar target se existir
    if y is not None:
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_processed = le_target.fit_transform(y.fillna('Unknown'))
            print(f"  🎯 Target codificado: {le_target.classes_}")
        else:
            y_processed = y.fillna(y.median())
            le_target = None
        
        print(f"\n📊 Classes do target: {pd.Series(y_processed).value_counts().to_dict()}")
    else:
        y_processed = None
        le_target = None
    
    # 7. Verificação final
    print(f"\n✅ Dataset final: {X_processed.shape}")
    print(f"✅ Valores nulos restantes: {X_processed.isnull().sum().sum()}")
    
    return X_processed, y_processed, label_encoders, le_target

def test_robust_model():
    """Teste do modelo com pré-processamento robusto"""
    print("🚀 TESTE DO MODELO ROBUSTO")
    
    # 1. Carregar dados
    train_df = pd.read_csv('data/raw/train.csv')
    
    # 2. Pré-processamento robusto
    X, y, encoders, target_encoder = robust_preprocessing(train_df)
    
    if y is None:
        print("❌ Não foi possível encontrar a coluna target")
        return False
    
    # 3. Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n📊 Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    
    # 4. Treinar modelo robusto (Random Forest lida bem com dados faltantes)
    print("\n🌲 Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=10,  # Evitar overfitting com poucos dados
        min_samples_split=5
    )
    
    rf.fit(X_train, y_train)
    
    # 5. Avaliar
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 RESULTADOS:")
    print(f"✅ Acurácia: {accuracy:.4f}")
    print(f"\n📋 Relatório de classificação:")
    print(classification_report(y_test, y_pred))
    
    # 6. Features importantes
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 10 FEATURES MAIS IMPORTANTES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n🎉 MODELO TREINADO COM SUCESSO!")
    return True

if __name__ == "__main__":
    success = test_robust_model()
    if success:
        print("\n✅ TESTE CONCLUÍDO - Modelo está funcionando!")
    else:
        print("\n❌ TESTE FALHOU - Verifique os dados")