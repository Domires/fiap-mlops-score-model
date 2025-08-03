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
    Função para pré-processar os dados de credit score
    
    Args:
        df (pd.DataFrame): DataFrame com os dados brutos
        is_training (bool): Se True, inclui a coluna target no processamento
        
    Returns:
        pd.DataFrame: DataFrame processado
    """
    # Cópia do dataframe
    df_processed = df.copy()
    
    # Removendo colunas desnecessárias
    columns_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Tratamento de valores problemáticos em colunas numéricas
    numeric_columns = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]
    
    for col in numeric_columns:
        if col in df_processed.columns:
            # Convertendo valores não numéricos para NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
            # Tratando outliers negativos para idade
            if col == 'Age':
                df_processed[col] = df_processed[col].apply(lambda x: np.nan if x < 0 or x > 100 else x)
    
    # Tratamento de valores categóricos problemáticos
    categorical_columns = [
        'Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 
        'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour'
    ]
    
    for col in categorical_columns:
        if col in df_processed.columns:
            # Substituindo valores problemáticos por NaN
            df_processed[col] = df_processed[col].replace(['_', '!@9#%8', '#F%$D@*&8', '_______', 'NA'], np.nan)
    
    return df_processed


def create_preprocessing_pipeline(X):
    """
    Cria um pipeline de pré-processamento para os dados
    
    Args:
        X (pd.DataFrame): DataFrame com as features
        
    Returns:
        ColumnTransformer: Pipeline de pré-processamento
    """
    # Identificando colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()
    
    # Pipeline de pré-processamento para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline de pré-processamento para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinando os pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def load_and_preprocess_data(train_path, test_path=None):
    """
    Carrega e pré-processa os dados de treino e teste
    
    Args:
        train_path (str): Caminho para o arquivo de treino
        test_path (str, optional): Caminho para o arquivo de teste
        
    Returns:
        tuple: (train_processed, test_processed, features)
    """
    # Carregando dados
    train_df = pd.read_csv(train_path)
    
    if test_path:
        test_df = pd.read_csv(test_path)
    else:
        test_df = None
    
    # Pré-processando dados
    train_processed = preprocess_credit_score_data(train_df, is_training=True)
    
    if test_df is not None:
        test_processed = preprocess_credit_score_data(test_df, is_training=False)
    else:
        test_processed = None
    
    # Extraindo lista de features
    features = list(train_processed.columns)
    if 'Credit_Score' in features:
        features.remove('Credit_Score')
    
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