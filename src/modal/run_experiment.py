#!/usr/bin/env python3
"""
Pipeline Modal - Experimento 02_xavier_rs_analysis.py
Configura e executa análise PREDEP no Modal Cloud
"""

import subprocess
import sys
from pathlib import Path


def check_modal():
    """Verifica instalação e autenticação do Modal."""
    print("🔍 Verificando Modal...")
    
    try:
        import modal
        print(f"   ✅ Modal v{modal.__version__}")
    except ImportError:
        print("   ❌ Modal não instalado")
        print("   Execute: pip install modal")
        return False
    
    result = subprocess.run("modal token new --help", shell=True, capture_output=True)
    if result.returncode != 0:
        print("   ❌ Modal CLI não configurado")
        print("   Execute: modal token new")
        return False
    
    print("   ✅ Modal configurado")
    return True


def check_data():
    """Verifica se dados mensais processados existem."""
    print("\n📂 Verificando dados...")
    
    base_dir = Path(__file__).parent.parent.parent
    required = [
        base_dir / "data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc",
        base_dir / "data/raw/mov_indices/AAO_indice data.csv"
    ]
    
    missing = []
    for file in required:
        if Path(file).exists():
            size_mb = Path(file).stat().st_size / (1024*1024)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file}")
            missing.append(file)
    
    if missing:
        print("\n   Execute primeiro: python tools/convert_xavier_daily_to_monthly.py")
        return False
    
    return True


def upload_to_modal():
    """Faz upload dos dados para Modal."""
    print("\n📤 Upload para Modal...")
    
    # Criar volume se não existir
    subprocess.run("modal volume create xavier-predep-data 2>/dev/null", shell=True)
    
    base_dir = Path(__file__).parent.parent.parent
    files = [
        (base_dir / "data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc", 
         "pr_monthly_RS.nc"),
        (base_dir / "data/raw/mov_indices/AAO_indice data.csv", 
         "AAO_indice_data.csv"),
        (base_dir / "src/predep_calculation/core.py", 
         "predep_core.py")
    ]
    
    for local, remote in files:
        if not Path(local).exists():
            print(f"   ❌ {local} não encontrado")
            return False
            
        print(f"   📤 {Path(local).name}...", end=" ")
        result = subprocess.run(
            f"modal volume put xavier-predep-data '{local}' '{remote}'",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0 or "already exists" in result.stderr:
            print("✅")
        else:
            print(f"❌ {result.stderr}")
    
    return True


def create_modal_script():
    """Cria script Modal adaptado do experimento."""
    print("\n📝 Criando script Modal...")
    
    modal_script = '''import modal

app = modal.App("xavier-predep-rs")
image = modal.Image.debian_slim().pip_install([
    "pandas", "xarray", "numpy", "matplotlib", "netcdf4", 
    "scipy", "astropy", "tqdm"
])
volume = modal.Volume.from_name("xavier-predep-data")

@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=32768,
    timeout=3600*8
)
def run_analysis():
    import sys
    sys.path.insert(0, "/data")
    
    # Importar função PREDEP
    exec(open("/data/predep_core.py").read(), globals())
    
    import pandas as pd
    import xarray as xr
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    
    print("="*60)
    print("ANÁLISE PREDEP - MODAL CLOUD")
    print("="*60)
    
    # Carregar dados
    ds = xr.open_dataset("/data/pr_monthly_RS.nc")
    mov_data = pd.read_csv("/data/AAO_indice_data.csv")
    mov_data['time'] = pd.to_datetime(mov_data['time'])
    mov_data['Year'] = mov_data['time'].dt.year
    mov_data['Month'] = mov_data['time'].dt.month
    mov_monthly = mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index()
    
    # Processar estações
    seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
    season_months = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
    
    results = {}
    for season, months in seasons.items():
        print(f"\\n📊 Processando {season}...")
        
        # Calcular anomalias sazonais
        mask = ds.time.dt.month.isin(months)
        season_subset = ds.pr.where(mask, drop=True)
        seasonal = season_subset.groupby(season_subset.time.dt.year).mean('time')
        anomalies = seasonal - seasonal.mean('year')
        
        # Calcular PREDEP para cada lag (0-11 = 12 lags)
        n_lags = 12
        alpha_results = np.full((n_lags, len(anomalies.latitude), len(anomalies.longitude)), np.nan)
        
        for lag in tqdm(range(n_lags), desc=f"{season} lags"):
            for i in range(len(anomalies.latitude)):
                for j in range(len(anomalies.longitude)):
                    precip = anomalies.isel(latitude=i, longitude=j).values
                    
                    if not np.all(np.isnan(precip)):
                        # Alinhar MoV
                        mov_aligned = []
                        precip_aligned = []
                        
                        for year_idx, year in enumerate(anomalies.year.values):
                            target_month = season_months[season] - lag
                            target_year = year
                            while target_month <= 0:
                                target_month += 12
                                target_year -= 1
                            
                            mov_row = mov_monthly[
                                (mov_monthly['Year'] == target_year) & 
                                (mov_monthly['Month'] == target_month)
                            ]
                            
                            if len(mov_row) == 1 and not np.isnan(precip[year_idx]):
                                mov_aligned.append(mov_row['value'].iloc[0])
                                precip_aligned.append(precip[year_idx])
                        
                        if len(mov_aligned) >= 10:
                            try:
                                alpha = predep(np.array(precip_aligned), np.array(mov_aligned), n_boot=100)
                                alpha_results[lag, i, j] = alpha
                            except:
                                pass
        
        results[season] = alpha_results
        print(f"   ✅ {season}: {np.sum(~np.isnan(alpha_results))} pontos válidos")
    
    # Salvar resultados
    output_dir = Path("/data/results")
    output_dir.mkdir(exist_ok=True)
    
    for season, data in results.items():
        ds_out = xr.Dataset(
            {f'alpha_lag{lag}': (['latitude', 'longitude'], data[lag]) 
             for lag in range(n_lags)},
            coords={'latitude': anomalies.latitude, 'longitude': anomalies.longitude}
        )
        ds_out.to_netcdf(output_dir / f"AAO_{season}_xavier_rs.nc")
    
    volume.commit()
    print("\\n✅ Análise concluída! Resultados salvos no volume Modal.")
    return "Success"

@app.local_entrypoint()
def main():
    result = run_analysis.remote()
    print(f"\\n🎉 {result}")
'''
    
    output_file = Path(__file__).parent / "modal_experiment.py"
    output_file.write_text(modal_script)
    print(f"   ✅ {output_file} criado")
    return True


def run_modal():
    """Executa análise no Modal."""
    print("\n🚀 Executando no Modal...")
    print("   (Isso pode levar várias horas)")
    
    script_path = Path(__file__).parent / "modal_experiment.py"
    result = subprocess.run(f"modal run {script_path}", shell=True)
    return result.returncode == 0


def download_results():
    """Baixa resultados do Modal."""
    print("\n📥 Baixando resultados...")
    
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "results/modal_xavier_rs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        f"modal volume get xavier-predep-data results/ {output_dir}/",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(f"   ✅ Resultados em: {output_dir.absolute()}")
        return True
    else:
        print(f"   ❌ Erro: {result.stderr}")
        return False


def main():
    """Pipeline principal."""
    print("="*70)
    print("PIPELINE MODAL - EXPERIMENTO 02_XAVIER_RS_ANALYSIS")
    print("="*70)
    
    # 1. Verificar Modal
    if not check_modal():
        return
    
    # 2. Verificar dados
    if not check_data():
        return
    
    # 3. Upload
    input("\n🔄 Pressione ENTER para fazer upload dos dados...")
    if not upload_to_modal():
        return
    
    # 4. Criar script
    if not create_modal_script():
        return
    
    # 5. Executar
    print("\n" + "="*70)
    print("⚠️  ATENÇÃO:")
    print("   - Análise pode levar 4-8 horas")
    print("   - Consumirá recursos do Modal (pode ter custo)")
    print("   - Processará 11.570 pontos × 4 estações × 12 lags (0-11)")
    print("="*70)
    
    response = input("\n❓ Executar no Modal? (s/N): ").lower()
    
    if response in ['s', 'sim', 'y', 'yes']:
        if run_modal():
            input("\n🔄 Pressione ENTER para baixar resultados...")
            download_results()
            print("\n🎉 Pipeline concluída!")
        else:
            print("\n❌ Erro na execução")
    else:
        print("\n⏸️  Cancelado")


if __name__ == "__main__":
    main()
