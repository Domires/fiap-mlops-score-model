"""
Módulo para pré-processamento dos dados de Credit Score.
Este módulo contém funções para limpar e preparar os dados para treinamento.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_credit_score_data(df, is_training=True):
    """
    Função para pré-processar os dados de credit score com tratamento robusto de valores nulos
    
    Args:
        df (pd.DataFrame): DataFrame com os dados brutos
        is_training (bool): Se True, inclui a coluna target no processamento
        
    Returns:
        pd.DataFrame: DataFrame processado
    """
    # Cópia do dataframe
    df_processed = df.copy()
    
    print(f"📊 Iniciando pré-processamento: {df_processed.shape}")
    
    # 1. Removendo colunas desnecessárias e com muitos nulos
    columns_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN']
    
    # Identificar colunas com mais de 70% de valores nulos
    null_percent = (df_processed.isnull().sum() / len(df_processed)) * 100
    high_null_cols = null_percent[null_percent > 70].index.tolist()
    columns_to_drop.extend(high_null_cols)
    
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    print(f"🗑️ Removidas {len([col for col in columns_to_drop if col in df_processed.columns])} colunas com muitos nulos")
    
    # 2. Separar colunas numéricas e categóricas para tratamento específico
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Remover target das listas se existir
    if 'Credit_Score' in numeric_columns:
        numeric_columns.remove('Credit_Score')
    if 'Credit_Score' in categorical_columns:
        categorical_columns.remove('Credit_Score')
    
    # 3. Tratamento robusto de colunas numéricas
    for col in numeric_columns:
        if col in df_processed.columns:
            # Converter para numérico
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
            # Imputar com mediana (mais robusto que média)
            median_val = df_processed[col].median()
            if pd.isna(median_val):  # Se todos são NaN, usar 0
                median_val = 0
            df_processed[col] = df_processed[col].fillna(median_val)
            
            # Tratamento especial para idade
            if col == 'Age':
                df_processed[col] = df_processed[col].apply(lambda x: 25 if x < 0 or x > 100 else x)
    
    # 4. Tratamento robusto de colunas categóricas
    for col in categorical_columns:
        if col in df_processed.columns:
            # Limpar valores problemáticos
            df_processed[col] = df_processed[col].replace(['_', '!@9#%8', '#F%$D@*&8', '_______', 'NA'], np.nan)
            
            # Imputar com moda ou valor padrão
            mode_values = df_processed[col].mode()
            if len(mode_values) > 0:
                fill_val = mode_values[0]
            else:
                fill_val = 'Unknown'
            df_processed[col] = df_processed[col].fillna(fill_val)
    
    print(f"✅ Pré-processamento concluído: {df_processed.shape}")
    print(f"✅ Valores nulos restantes: {df_processed.isnull().sum().sum()}")
    
    return df_processed


def create_preprocessing_pipeline(X):
    """
    Cria um pipeline de pré-processamento robusto para os dados
    
    Args:
        X (pd.DataFrame): DataFrame com as features
        
    Returns:
        ColumnTransformer: Pipeline de pré-processamento
    """
    print(f"🔧 Criando pipeline para {X.shape[1]} features")
    
    # Identificando colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()
    
    print(f"📊 Features numéricas: {len(numeric_features)}")
    print(f"📊 Features categóricas: {len(categorical_features)}")
    
    # Pipeline robusto para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Mediana é mais robusta
        ('scaler', StandardScaler())
    ])
    
    # Pipeline robusto para features categóricas  
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))  # drop='first' evita multicolinearidade
    ])
    
    # Lista de transformadores
    transformers = []
    
    if len(numeric_features) > 0:
        transformers.append(('num', numeric_transformer, numeric_features))
        print(f"✅ Pipeline numérico criado para: {numeric_features}")
    
    if len(categorical_features) > 0:
        transformers.append(('cat', categorical_transformer, categorical_features))
        print(f"✅ Pipeline categórico criado para: {categorical_features}")
    
    if len(transformers) == 0:
        raise ValueError("Nenhuma feature válida encontrada para criar o pipeline!")
    
    # Combinando os pipelines
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Remove colunas não especificadas
    )
    
    print(f"🎯 Pipeline de pré-processamento criado com sucesso!")
    return preprocessor


def load_and_preprocess_data(train_path, test_path=None):
    """
    Carrega e pré-processa os dados de treino e teste com tratamento robusto
    
    Args:
        train_path (str): Caminho para o arquivo de treino
        test_path (str, optional): Caminho para o arquivo de teste
        
    Returns:
        tuple: (train_processed, test_processed, features)
    """
    print(f"📂 Carregando dados de: {train_path}")
    
    # Carregando dados
    train_df = pd.read_csv(train_path)
    
    if test_path:
        test_df = pd.read_csv(test_path)
        print(f"📂 Carregando dados de teste de: {test_path}")
    else:
        test_df = None
    
    # Pré-processando dados
    print("\n🔄 Pré-processando dados de treino...")
    train_processed = preprocess_credit_score_data(train_df, is_training=True)
    
    # Tratamento robusto do target (Credit_Score)
    if 'Credit_Score' in train_processed.columns:
        # Verificar se há valores nulos no target
        null_count = train_processed['Credit_Score'].isnull().sum()
        if null_count > 0:
            print(f"⚠️ Encontrados {null_count} valores nulos no target")
            # Remover linhas com target nulo (dados inválidos para treinamento)
            train_processed = train_processed.dropna(subset=['Credit_Score'])
            print(f"🗑️ Removidas {null_count} linhas com target nulo")
            print(f"📊 Dataset de treino final: {train_processed.shape}")
    
    if test_df is not None:
        print("\n🔄 Pré-processando dados de teste...")
        test_processed = preprocess_credit_score_data(test_df, is_training=False)
    else:
        test_processed = None
    
    # Extraindo lista de features (sem o target)
    features = list(train_processed.columns)
    if 'Credit_Score' in features:
        features.remove('Credit_Score')
    
    print(f"\n✅ Features disponíveis: {len(features)}")
    print(f"🎯 Target: Credit_Score")
    
    return train_processed, test_processed, features


if __name__ == "__main__":
    # Teste do módulo
    train_processed, test_processed, features = load_and_preprocess_data(
        'data/raw/train.csv',
        'data/raw/test.csv'
    )
    
    print(f"Dados de treino processados: {train_processed.shape}")
    if test_processed is not None:
        print(f"Dados de teste processados: {test_processed.shape}")
    print(f"Número de features: {len(features)}")
    print(f"Features: {features}")