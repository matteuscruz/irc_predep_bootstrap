#!/usr/bin/env python3
"""
Script de Monitoramento - Execução com Métricas
Monitora CPU, memória e progresso em tempo real
"""

import subprocess
import psutil
import time
from datetime import datetime
from pathlib import Path
import sys

def get_system_info():
    """Retorna informações do sistema."""
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    
    return {
        'cpu_physical': cpu_count,
        'cpu_logical': cpu_count_logical,
        'mem_total_gb': mem.total / (1024**3),
        'mem_available_gb': mem.available / (1024**3)
    }


def monitor_execution(script_path, check_interval=30):
    """
    Executa script e monitora recursos.
    
    Parameters
    ----------
    script_path : str
        Caminho do script Python a executar
    check_interval : int
        Intervalo de checagem em segundos
    """
    print("=" * 80)
    print("MONITOR DE EXECUÇÃO - ALTA PERFORMANCE")
    print("=" * 80)
    print(f"Script: {script_path}")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Info do sistema
    sys_info = get_system_info()
    print("SISTEMA:")
    print(f"  CPUs físicas: {sys_info['cpu_physical']}")
    print(f"  CPUs lógicas: {sys_info['cpu_logical']}")
    print(f"  RAM total: {sys_info['mem_total_gb']:.1f} GB")
    print(f"  RAM disponível: {sys_info['mem_available_gb']:.1f} GB")
    print()
    print("=" * 80)
    print()
    
    # Iniciar processo
    start_time = time.time()
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitorar
    try:
        while True:
            # Ler output
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Checar se terminou
            if process.poll() is not None:
                break
            
            # Métricas a cada intervalo
            elapsed = time.time() - start_time
            if int(elapsed) % check_interval == 0 and elapsed > 0:
                cpu_percent = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                
                print()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Tempo: {elapsed/60:.1f}min | "
                      f"CPU: {cpu_percent:.1f}% | "
                      f"RAM: {mem.percent:.1f}% ({mem.used/(1024**3):.1f}GB)")
                print()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
        process.terminate()
        return False
    
    # Ler output restante
    for output in process.stdout:
        print(output.strip())
    
    # Resultado final
    elapsed_total = time.time() - start_time
    print()
    print("=" * 80)
    
    if process.returncode == 0:
        print("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO")
    else:
        print(f"❌ EXECUÇÃO FALHOU (código: {process.returncode})")
    
    print(f"Tempo total: {elapsed_total/60:.1f} min ({elapsed_total:.1f}s)")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return process.returncode == 0


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print("Uso: python monitor_execution.py <script.py>")
        print()
        print("Exemplos:")
        print("  python monitor_execution.py experiments/high_performance_test.py")
        print("  python monitor_execution.py experiments/full_analysis_optimized.py")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    if not Path(script_path).exists():
        print(f"❌ Erro: Script não encontrado: {script_path}")
        sys.exit(1)
    
    success = monitor_execution(script_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
