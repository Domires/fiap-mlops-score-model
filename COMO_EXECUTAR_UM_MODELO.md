# 🎯 COMO EXECUTAR APENAS 1 MODELO (RANDOM FOREST)

## ❌ PROBLEMA IDENTIFICADO

Você estava executando **`train_credit_score_model.py`** que chama `src/model_training.py` e treina **4 modelos**:
- ❌ Logistic Regression
- ❌ Random Forest  
- ❌ XGBoost
- ❌ LightGBM

## ✅ 3 SOLUÇÕES DISPONÍVEIS

### 🚀 **OPÇÃO 1: SCRIPT SIMPLES (RECOMENDADO)**
```bash
python simple_credit_score_model.py
```
**Vantagens:**
- ✅ Código mais limpo e robusto
- ✅ Sem MLflow (evita problemas de endpoint)
- ✅ Tratamento automático de encoding
- ✅ Visualizações automáticas
- ✅ Documentação completa

---

### 🔧 **OPÇÃO 2: SCRIPT ORIGINAL MODIFICADO**
```bash
python train_credit_score_model.py
```
**O que mudou:**
- ✅ Agora treina apenas Random Forest
- ✅ Sem MLflow para evitar erros
- ✅ Usa a nova função `main_random_forest_only()`
- ⚠️ Pedirá confirmação antes de executar

---

### 📝 **OPÇÃO 3: EXECUTAR FUNÇÃO ESPECÍFICA**
```python
from src.model_training import main_random_forest_only
main_random_forest_only()
```

---

## 🎯 COMPARAÇÃO DAS OPÇÕES

| Característica | Opção 1 (Simples) | Opção 2 (Modificado) | Opção 3 (Função) |
|---|---|---|---|
| **Facilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Robustez** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Visualizações** | ✅ Automático | ❌ Manual | ❌ Manual |
| **Documentação** | ✅ Completa | ⭐⭐ | ⭐ |
| **Tratamento Erros** | ✅ Robusto | ⭐⭐⭐ | ⭐⭐ |

---

## 🏆 **RECOMENDAÇÃO FINAL**

### Use a **OPÇÃO 1**:
```bash
python simple_credit_score_model.py
```

**Por quê?**
- 🎯 **Mais simples**: Código dedicado apenas ao Random Forest
- 🔧 **Mais robusto**: Tratamento completo de erros e edge cases
- 📊 **Visualizações**: Gráficos automáticos de matriz de confusão e feature importance
- 💾 **Salvamento completo**: Modelo + encoder + predições + info
- 📝 **Bem documentado**: Output detalhado e explicativo

---

## 📁 ARQUIVOS GERADOS (TODAS AS OPÇÕES)

```
models/
├── random_forest_credit_score.pkl  # Modelo treinado
├── label_encoder.pkl              # Conversor de classes (se necessário)
├── predictions.csv                # Predições no teste
├── model_info.json               # Informações do modelo
├── confusion_matrix.png          # Matriz de confusão (Opção 1)
└── feature_importance.png        # Importância das features (Opção 1)
```

---

## 🔥 **EXECUTE AGORA**

```bash
# RECOMENDADO
python simple_credit_score_model.py

# OU (se preferir usar os arquivos originais modificados)
python train_credit_score_model.py
```

**✅ Qualquer uma das opções vai treinar APENAS 1 modelo Random Forest!**

---

## ⚠️ IMPORTANTE

- **NÃO execute mais o código antigo** que treina múltiplos modelos
- **Use sempre uma das 3 opções acima**
- **Se ainda aparecer múltiplos modelos**, você está executando algum código diferente

---

## 🆘 SUPORTE

Se ainda tiver problemas:
1. Verifique qual arquivo está executando
2. Use a **Opção 1** que é mais robusta
3. Verifique se não há outros scripts sendo executados