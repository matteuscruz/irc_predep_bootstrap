#!/usr/bin/env python3
"""
Teste Modal - Verificação de funcionamento básico
Execute este script primeiro para verificar se Modal está configurado corretamente
"""

import modal

# Imagem mínima para teste
test_image = modal.Image.debian_slim().pip_install(["numpy", "pandas", "tqdm"])

app = modal.App("test-modal-predep", image=test_image)

@app.function(timeout=300)  # 5 minutos timeout
def test_basic_functionality():
    """Testa funcionalidades básicas no Modal."""
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import time
    
    print("🧪 TESTE MODAL - PREDEP ANALYSIS")
    print("="*50)
    
    # Teste 1: NumPy
    print("📊 Teste 1: NumPy...")
    data = np.random.normal(0, 1, (1000, 100))
    mean_val = np.mean(data)
    print(f"   ✅ NumPy OK - Média: {mean_val:.4f}")
    
    # Teste 2: Pandas
    print("📊 Teste 2: Pandas...")
    df = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'C': pd.date_range('2020-01-01', periods=1000, freq='D')
    })
    corr = df['A'].corr(df['B'])
    print(f"   ✅ Pandas OK - Correlação A-B: {corr:.4f}")
    
    # Teste 3: Progress bar
    print("📊 Teste 3: Progress bar...")
    for i in tqdm(range(100), desc="Teste progress", unit="step"):
        time.sleep(0.01)  # Simula processamento
    print("   ✅ TQDM OK")
    
    # Teste 4: Simulação de cálculo PREDEP
    print("📊 Teste 4: Simulação PREDEP...")
    
    def simple_bootstrap_test(x, y, n_boot=100):
        """Simulação simples de bootstrap para teste."""
        correlations = []
        n = len(x)
        
        for _ in tqdm(range(n_boot), desc="Bootstrap", leave=False):
            indices = np.random.choice(n, n, replace=True)
            boot_x = x[indices]
            boot_y = y[indices]
            
            if np.var(boot_x) > 0 and np.var(boot_y) > 0:
                corr = np.corrcoef(boot_x, boot_y)[0, 1]
                correlations.append(abs(corr))
        
        return np.percentile(correlations, 95) if correlations else np.nan
    
    # Gerar dados de teste
    n_points = 500
    x = np.random.normal(0, 1, n_points)
    y = 0.3 * x + np.random.normal(0, 1, n_points)  # Correlação ~0.3
    
    alpha = simple_bootstrap_test(x, y, n_boot=200)
    print(f"   ✅ Bootstrap PREDEP OK - α: {alpha:.4f}")
    
    # Teste 5: Simulação de processamento espacial
    print("📊 Teste 5: Processamento espacial...")
    grid_size = 20
    results = np.zeros((grid_size, grid_size))
    
    with tqdm(total=grid_size*grid_size, desc="Grid espacial") as pbar:
        for i in range(grid_size):
            for j in range(grid_size):
                # Simula cálculo PREDEP para cada ponto
                x_sim = np.random.normal(0, 1, 50)
                y_sim = np.random.normal(0, 1, 50) 
                results[i, j] = simple_bootstrap_test(x_sim, y_sim, n_boot=50)
                pbar.update(1)
    
    valid_results = np.sum(~np.isnan(results))
    print(f"   ✅ Grid espacial OK - {valid_results}/{grid_size*grid_size} pontos válidos")
    
    # Resumo final
    print("\n" + "="*50)
    print("🎉 TODOS OS TESTES PASSARAM!")
    print("✅ Modal está funcionando corretamente")
    print("✅ Dependências instaladas")
    print("✅ Progress bars funcionando") 
    print("✅ Bootstrap PREDEP simulado OK")
    print("✅ Processamento espacial OK")
    print("\n🚀 Pronto para executar análise PREDEP completa!")
    
    return {
        "status": "success",
        "tests_passed": 5,
        "sample_alpha": float(alpha),
        "valid_grid_points": int(valid_results),
        "grid_total": grid_size*grid_size
    }


@app.function(timeout=60)
def test_environment_info():
    """Mostra informações do ambiente Modal."""
    import platform
    import sys
    
    print("🖥️ INFORMAÇÕES DO AMBIENTE MODAL")
    print("="*40)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Testar recursos disponíveis
    try:
        import psutil
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        print("psutil não disponível - informações limitadas")
    
    return {
        "python_version": sys.version,
        "platform": platform.platform()
    }


@app.local_entrypoint() 
def main():
    """Executa todos os testes."""
    print("🧪 Iniciando testes Modal...")
    print("📡 Enviando para infraestrutura Modal...\n")
    
    # Teste ambiente
    env_info = test_environment_info.remote()
    print(f"\n📋 Ambiente: {env_info}")
    
    # Teste funcionalidade
    test_result = test_basic_functionality.remote()
    
    print(f"\n📊 Resultado dos testes: {test_result}")
    
    if test_result["status"] == "success":
        print("\n🎉 MODAL CONFIGURADO CORRETAMENTE!")
        print("Agora você pode executar:")
        print("modal run experiments/modal_xavier_rs_analysis.py")
    else:
        print("\n❌ Alguns testes falharam. Verifique a configuração.")
    
    return test_result


if __name__ == "__main__":
    # Para executar: modal run experiments/test_modal.py
    pass