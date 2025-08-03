Trabalho de MLOPS:

Contexto: A QuantumFinance realizou uma parceria com algumas empresas clientes para fornecer o score de crédito de seus clientes optantes para permitir melhores condições de pagamento.
Para que o modelo seja mais simples e diferente do modelo de score principal do banco, será necessário treinar esse modelo com os dados das transações mais recentes dos clientes.

Entrega: Para permitir governança interna e integração com outros sistemas, esta solução precisa incluir:

- Template de repositório para organização dos arquivos (dataset, notebook, modelo, etc.)
- Rastreamento dos experimentos do treinamento do modelo
- Versionamento do modelo
- Disponibilização de endpoint de API seguro com autenticação e throttling
- Documentação da API
- Aplicação do modelo no Streamlit com a API disponibilizada

Repositórios de referência para consulta:
- https://github.com/michelpf/fiap-ds-mlops-10dtsr-laptop-pricing-brl
- https://github.com/michelpf/fiap-ds-mlops-10dtsr-api-laptop-pricing-brl
- https://github.com/michelpf/fiap-ds-mlops-10dtsr-app-laptop-pricing-brl
- https://github.com/michelpf/fiap-ds-mlops-laptop-pricing-model-drift

Observações:
- Por enquanto iremos focar somente no modelo, e não no front-end e nem no back-end (API)
- O Modelo será um modelo de classificação de score de crédito
- Existem arquivos com prefixo "ref_", isso significa que são somente arquivos que foram usados para outro propósito, mas podem servir como referência para a construção desse trabalho
- Na pasta references existem arquivos de referência para construção do trabalho 

Dataset à ser utilizado no trabalho de MLOPS:
- https://www.kaggle.com/datasets/parisrohan/credit-score-classification
- Amostra de dados de treinamento e teste desse modelo estão dentro da pasta references (exemplo_train.csv e exemplo_test.csv)

Dicionário de Dados do dataset:
- Customer_ID
Represents a unique identification of a person
Month
- Represents the month of the year
Name
- Represents the name of a person 
Age
- Represents the age of the person
SSN
- Represents the social security number of a person
Occupation
- Represents the occupation of the person
Annual_Income
- Represents the annual income of the person
Monthly_Inhand_Salary
- Represents the monthly base salary of a person
Num_Bank_Accounts
- Represents the number of bank accounts a person holds   
Num_Credit_Card
- Represents the number of other credit cards held by a person
Interest_Rate
- Represents the interest rate on credit card
Num_of_Loan
- Represents the number of loans taken from the bank
Type_of_Loan
- Represents the types of loan taken by a person
Delay_from_due_date
- Represents the average number of days delayed from the payment date
Num_of_Delayed_Payment
- Represents the average number of payments delayed by a person   
Changed_Credit_Limit
- Represents the percentage change in credit card limit
Num_Credit_Inquiries
- Represents the number of credit card inquiries
Credit_Mix
- Represents the classification of the mix of credits
Outstanding_Debt
- Represents the remaining debt to be paid (in USD)
Credit_Utilization_Ratio
- Represents the utilization ratio of credit card
Credit_History_Age
- Represents the age of credit history of the person
Payment_of_Min_Amount
- Represents whether only the minimum amount was paid by the person
Total_EMI_per_month
- Represents the monthly EMI payments (in USD)
Amount_invested_monthly
- Represents the monthly amount invested by the customer (in USD)
Payment_Behaviour
- Represents the payment behavior of the customer (in USD)
Monthly_Balance
- Represents the monthly balance amount of the customer (in USD)
Credit_Score
- Represents the bracket of credit score (Poor, Standard, Good)