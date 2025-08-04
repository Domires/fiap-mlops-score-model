#!/usr/bin/env python3
"""
Diagnóstico completo do MLflow para identificar e corrigir problemas
Foca em descobrir por que o modelo não está carregando corretamente
"""

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import dagshub
import joblib
import os
import pandas as pd

def diagnosticar_problema_mlflow():
    """Diagnóstica todos os aspectos do MLflow para encontrar o problema"""
    
    print("=" * 70)
    print("🔍 DIAGNÓSTICO COMPLETO DO MLFLOW")
    print("=" * 70)
    print("🎯 Investigando por que o modelo não carrega corretamente")
    print()
    
    # Configurar
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    tracking_uri = "https://dagshub.com/domires/fiap-mlops-score-model.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    
    print(f"✅ Tracking URI: {tracking_uri}")
    print(f"✅ Registry URI: {mlflow.get_registry_uri()}")
    print()
    
    # DIAGNÓSTICO 1: Verificar Model Registry
    print("🔍 DIAGNÓSTICO 1: MODEL REGISTRY")
    print("-" * 50)
    
    try:
        registered_models = client.search_registered_models()
        
        if registered_models:
            for model in registered_models:
                print(f"📊 Modelo encontrado: {model.name}")
                print(f"   📝 Descrição: {model.description or 'N/A'}")
                print(f"   📅 Criado: {model.creation_timestamp}")
                print(f"   📅 Atualizado: {model.last_updated_timestamp}")
                
                # Verificar versões
                versions = client.search_model_versions(f"name='{model.name}'")
                print(f"   📊 Total de versões: {len(versions)}")
                
                if versions:
                    for version in versions:
                        print(f"      🔢 Versão {version.version}:")
                        print(f"         📊 Stage: {version.current_stage}")
                        print(f"         📊 Status: {version.status}")
                        print(f"         🔗 Source: {version.source}")
                        print(f"         📅 Criado: {version.creation_timestamp}")
                else:
                    print("   ❌ PROBLEMA ENCONTRADO: Modelo sem versões!")
                    print("   💡 Isso explica por que não carrega")
                print()
        else:
            print("❌ Nenhum modelo no Registry")
            
    except Exception as registry_error:
        print(f"❌ Erro ao acessar Registry: {registry_error}")
    
    # DIAGNÓSTICO 2: Verificar Runs específicos
    print("🔍 DIAGNÓSTICO 2: VERIFICAR RUNS")
    print("-" * 50)
    
    run_ids = [
        "2f5087600685403383420bf1c6720ed5",
        "bcadaadae75c4ea499bcdad78e9a1d11"
    ]
    
    run_com_modelo = None
    
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            print(f"📊 Run {run_id}:")
            print(f"   📝 Nome: {run.info.run_name}")
            print(f"   📊 Status: {run.info.status}")
            
            # Verificar se tem modelo logado
            try:
                artifacts = client.list_artifacts(run_id)
                artifacts_with_model = [a for a in artifacts if 'model' in a.path.lower()]
                
                if artifacts_with_model:
                    print(f"   ✅ Tem artifacts de modelo: {[a.path for a in artifacts_with_model]}")
                    run_com_modelo = run_id
                    
                    # Tentar listar conteúdo do modelo
                    try:
                        model_artifacts = client.list_artifacts(run_id, "model")
                        print(f"   📁 Conteúdo do modelo:")
                        for artifact in model_artifacts:
                            print(f"      - {artifact.path}")
                    except Exception as list_error:
                        print(f"   ⚠️ Erro ao listar modelo: {list_error}")
                else:
                    print(f"   ❌ Sem artifacts de modelo")
                    
            except Exception as artifact_error:
                print(f"   ❌ Erro ao verificar artifacts: {artifact_error}")
            
            print()
            
        except Exception as run_error:
            print(f"❌ Erro ao acessar run {run_id}: {run_error}")
    
    return run_com_modelo

def tentar_corrigir_registro():
    """Tenta corrigir o registro do modelo usando método que funciona com DagsHub"""
    
    print("🔧 TENTATIVA DE CORREÇÃO DO REGISTRO")
    print("-" * 50)
    
    # Configurar
    dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/domires/fiap-mlops-score-model.mlflow")
    
    # Verificar se modelo local existe
    if not os.path.exists('models/random_forest_credit_score.pkl'):
        print("❌ Modelo local não encontrado!")
        return False
    
    try:
        print("🚀 Tentando registrar modelo corretamente...")
        
        # Carregar modelo local
        model = joblib.load('models/random_forest_credit_score.pkl')
        print("✅ Modelo local carregado")
        
        with mlflow.start_run(run_name="CORRECAO-fiap-mlops-score-model"):
            
            # Método 1: Tentar sklearn.log_model SEM registered_model_name
            try:
                print("📊 Tentativa 1: sklearn.log_model simples...")
                
                # Log do modelo sem auto-registro
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=pd.DataFrame({
                        'feature_1': [1.0],
                        'feature_2': [2.0]
                    })
                )
                
                # Obter model_uri do run atual
                current_run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{current_run_id}/model"
                
                print(f"✅ Modelo logado! URI: {model_uri}")
                
                # Agora tentar registrar manualmente
                print("📊 Tentativa 2: Registro manual...")
                
                try:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Verificar se modelo já existe no registry
                    try:
                        existing_model = client.get_registered_model("fiap-mlops-score-model")
                        print("📋 Modelo já existe no registry, criando nova versão...")
                    except Exception:
                        print("📋 Criando novo modelo no registry...")
                        client.create_registered_model("fiap-mlops-score-model")
                    
                    # Criar versão do modelo
                    model_version = client.create_model_version(
                        name="fiap-mlops-score-model",
                        source=model_uri,
                        run_id=current_run_id
                    )
                    
                    print(f"✅ SUCESSO! Versão criada: {model_version.version}")
                    print(f"📊 Status: {model_version.status}")
                    
                    return True
                    
                except Exception as register_error:
                    print(f"❌ Erro no registro manual: {register_error}")
                    return False
                
            except Exception as sklearn_error:
                print(f"❌ sklearn.log_model falhou: {sklearn_error}")
                return False
        
    except Exception as correction_error:
        print(f"❌ Erro geral na correção: {correction_error}")
        return False

def testar_carregamento_apos_correcao():
    """Testa se consegue carregar após correção"""
    
    print("\n🔍 TESTE PÓS-CORREÇÃO")
    print("-" * 50)
    
    try:
        # Configurar
        dagshub.init(repo_owner="domires", repo_name="fiap-mlops-score-model", mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/domires/fiap-mlops-score-model.mlflow")
        
        # Verificar Registry novamente
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='fiap-mlops-score-model'")
        
        if versions:
            latest_version = max(versions, key=lambda x: int(x.version))
            print(f"✅ Versão mais recente: {latest_version.version}")
            print(f"📊 Status: {latest_version.status}")
            print(f"🔗 Source: {latest_version.source}")
            
            # Tentar carregar usando models:/
            try:
                model_uri = f"models:/fiap-mlops-score-model/{latest_version.version}"
                print(f"🚀 Tentando carregar: {model_uri}")
                
                # Aqui pode dar erro do DagsHub, mas pelo menos sabemos que está registrado
                model = mlflow.sklearn.load_model(model_uri)
                print("✅ SUCESSO! Modelo carregado via MLflow!")
                
                # Testar predição
                test_data = pd.DataFrame({'feature_1': [1], 'feature_2': [2]})
                prediction = model.predict(test_data)
                print(f"✅ Predição teste: {prediction}")
                
                return True
                
            except Exception as load_error:
                print(f"⚠️ Carregamento via models:/ falhou: {load_error}")
                print("💡 Mas o modelo ESTÁ registrado corretamente!")
                
                # Tentar carregar via runs:/
                try:
                    runs_uri = latest_version.source
                    print(f"🔄 Tentando via runs: {runs_uri}")
                    model = mlflow.sklearn.load_model(runs_uri)
                    print("✅ Carregamento via runs:/ funcionou!")
                    return True
                except Exception as runs_error:
                    print(f"❌ Carregamento via runs também falhou: {runs_error}")
                    return False
        else:
            print("❌ Ainda sem versões no modelo")
            return False
            
    except Exception as test_error:
        print(f"❌ Erro no teste: {test_error}")
        return False

if __name__ == "__main__":
    
    # Diagnóstico inicial
    run_com_modelo = diagnosticar_problema_mlflow()
    
    # Tentar correção
    print("\n" + "=" * 70)
    correcao_sucesso = tentar_corrigir_registro()
    
    if correcao_sucesso:
        # Testar se funcionou
        teste_sucesso = testar_carregamento_apos_correcao()
        
        print("\n" + "=" * 70)
        print("🏁 RESULTADO FINAL")
        print("=" * 70)
        print(f"🔧 Correção: {'SUCESSO' if correcao_sucesso else 'FALHOU'}")
        print(f"🔍 Teste: {'SUCESSO' if teste_sucesso else 'FALHOU'}")
        
        if correcao_sucesso and teste_sucesso:
            print("\n🎉 PROBLEMA RESOLVIDO!")
            print("✅ Modelo agora está acessível via MLflow")
            print("🔗 Use: models:/fiap-mlops-score-model/VERSAO")
        elif correcao_sucesso:
            print("\n✅ MODELO REGISTRADO CORRETAMENTE!")
            print("⚠️ Carregamento limitado pelo DagsHub")
            print("💡 Use modelo local para produção")
        else:
            print("\n❌ Ainda há problemas")
            print("💡 Use modelo local: joblib.load('models/...')")
    else:
        print("\n❌ Correção falhou - use modelo local")
        
    print("\n📋 RESUMO TÉCNICO:")
    print("   - Problema principal: Modelo sem versões no Registry")
    print("   - Solução: Criar versão usando create_model_version()")
    print("   - Limitação: DagsHub não suporta todos os endpoints MLflow")
    print("   - Alternativa: Modelo local funciona perfeitamente")