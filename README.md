# FIAP MLOps - Modelo de Classificação de Score de Crédito

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este projeto implementa um sistema completo de MLOps para classificação de score de crédito, desenvolvido para a QuantumFinance como parte de uma parceria para fornecer scores de crédito aos clientes.

## Sobre o Projeto

O modelo classifica clientes em três categorias de score de crédito:
- **Good**: Bom score de crédito
- **Standard**: Score de crédito padrão  
- **Poor**: Score de crédito ruim

### Dataset
O projeto utiliza o dataset [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) do Kaggle, que contém informações financeiras e comportamentais dos clientes.

### Características do Sistema
- **Rastreamento de Experimentos**: Uso do MLflow para tracking
- **Versionamento de Modelos**: Controle de versões dos modelos treinados
- **Pipeline de Dados**: Pré-processamento automatizado dos dados
- **Múltiplos Algoritmos**: Comparação entre diferentes modelos de classificação
- **Preparado para Produção**: Estrutura pronta para integração com API

## Quick Start

### 1. Treinamento de Modelos

```bash
# Execute o script principal de treinamento
python train_credit_score_model.py
```

Este script irá:
- Carregar e pré-processar os dados de exemplo
- Treinar múltiplos modelos (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Registrar todos os experimentos no MLflow
- Comparar as métricas de performance

### 2. Notebook de Desenvolvimento

```bash
# Abra o notebook para exploração interativa
jupyter notebook notebooks/credit_score_model_development.ipynb
```

### 3. Visualização dos Resultados

Acesse o MLflow UI para comparar os modelos:
```bash
mlflow ui
```

## Estrutura dos Dados

### Features Principais
- **Dados Demográficos**: Age, Occupation
- **Dados Financeiros**: Annual_Income, Monthly_Inhand_Salary, Outstanding_Debt
- **Comportamento de Crédito**: Credit_Mix, Payment_Behaviour, Credit_Utilization_Ratio
- **Histórico**: Credit_History_Age, Num_of_Delayed_Payment

### Target
- **Credit_Score**: Good, Standard, Poor

## Modelos Implementados

1. **Logistic Regression**: Modelo linear baseline
2. **Random Forest**: Ensemble de árvores de decisão
3. **XGBoost**: Gradient boosting otimizado
4. **LightGBM**: Gradient boosting eficiente

### Métricas de Avaliação
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- AUC-ROC (multiclass)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fiap-mlops-score-model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fiap-mlops-score-model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fiap-mlops-score-model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

