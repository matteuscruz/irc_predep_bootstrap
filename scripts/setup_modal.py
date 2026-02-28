#!/usr/bin/env python3
"""
Automação completa para executar análise PREDEP no Modal
Este script automatiza todo o processo de setup e execução
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(command, description):
    """Executa comando e mostra progresso."""
    print(f"\n🔄 {description}...")
    print(f"Executando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Sucesso!")
            if result.stdout.strip():
                print("Saída:", result.stdout.strip())
            return True
        else:
            print(f"❌ {description} - Erro!")
            if result.stderr.strip():
                print("Erro:", result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ {description} - Exceção: {e}")
        return False

def check_modal_installed():
    """Verifica se Modal está instalado."""
    try:
        import modal
        return True
    except ImportError:
        return False

def check_modal_authenticated():
    """Verifica se Modal está autenticado."""
    result = subprocess.run("modal token current 2>/dev/null", shell=True, capture_output=True)
    return result.returncode == 0

def main():
    """Processo completo de setup e execução."""
    print("🚀 AUTOMAÇÃO MODAL - PREDEP ANALYSIS")
    print("="*50)
    print("Este script irá:")
    print("1. Verificar/instalar Modal")
    print("2. Fazer login no Modal")  
    print("3. Executar testes")
    print("4. Executar análise PREDEP")
    print("="*50)
    
    # Verificar diretório atual
    if not Path("experiments").exists():
        print("❌ Execute este script do diretório raiz do projeto!")
        sys.exit(1)
    
    # Passo 1: Verificar/instalar Modal
    if not check_modal_installed():
        print("\n📦 Modal não encontrado. Instalando...")
        if not run_command("pip install modal", "Instalação Modal"):
            print("❌ Falha na instalação do Modal. Execute manualmente: pip install modal")
            sys.exit(1)
    else:
        print("✅ Modal já está instalado!")
    
    # Passo 2: Verificar autenticação
    if not check_modal_authenticated():
        print("\n🔐 Modal não está autenticado.")
        print("Por favor, execute o comando abaixo e siga as instruções:")
        print("modal token new")
        print("\nApós autenticar, execute este script novamente.")
        
        # Tentar executar automaticamente
        choice = input("\nTentar autenticar automaticamente? (y/N): ").lower()
        if choice == 'y':
            if not run_command("modal token new", "Autenticação Modal"):
                print("❌ Falha na autenticação. Execute manualmente: modal token new")
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        print("✅ Modal já está autenticado!")
    
    # Passo 3: Executar testes
    print("\n🧪 Executando testes Modal...")
    if not run_command("modal run experiments/test_modal.py", "Testes Modal"):
        print("❌ Testes falharam. Verifique a configuração.")
        choice = input("Continuar mesmo assim? (y/N): ").lower()
        if choice != 'y':
            sys.exit(1)
    
    # Passo 4: Perguntar sobre execução da análise
    print("\n" + "="*50)
    print("🎯 PRONTO PARA ANÁLISE PREDEP!")
    print("="*50)
    
    print("\nOpções disponíveis:")
    print("1. Análise rápida (dados simulados, ~5 min, ~$1)")
    print("2. Análise com seus dados (configure primeiro)")
    print("3. Pular análise (apenas setup)")
    
    choice = input("\nEscolha uma opção (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Executando análise rápida com dados simulados...")
        print("⏱️  Tempo estimado: 5-10 minutos")
        print("💰 Custo estimado: $1-3")
        
        confirm = input("Confirma execução? (y/N): ").lower()
        if confirm == 'y':
            success = run_command(
                "modal run experiments/modal_xavier_rs_analysis.py", 
                "Análise PREDEP Modal"
            )
            
            if success:
                print("\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
                print("\n📥 Para baixar resultados:")
                print("modal volume get xavier-data results/")
            else:
                print("\n❌ Análise falhou. Verifique os logs acima.")
    
    elif choice == "2":
        print("\n📋 Para usar seus dados reais:")
        print("1. Edite 'experiments/modal_xavier_rs_analysis.py'")
        print("2. Substitua as funções de carregamento simulado")
        print("3. Use exemplos em 'experiments/modal_setup_guide.py'")
        print("4. Execute: modal run experiments/modal_xavier_rs_analysis.py")
        
    elif choice == "3":
        print("\n✅ Setup concluído! Para executar análise:")
        print("modal run experiments/modal_xavier_rs_analysis.py")
        
    else:
        print("❌ Opção inválida.")
        sys.exit(1)
    
    # Informações finais
    print("\n" + "="*50)
    print("📚 RECURSOS ÚTEIS:")
    print("="*50)
    print("• Logs em tempo real: na execução do modal run")
    print("• Interface web: https://modal.com/apps")
    print("• Listar volumes: modal volume ls")
    print("• Baixar resultados: modal volume get xavier-data")
    print("• Documentação: README_MODAL.md")
    print("• Testes: modal run experiments/test_modal.py")
    
    print(f"\n🎯 SETUP MODAL CONCLUÍDO!")
    print("Agora você pode executar análises PREDEP na nuvem! 🚀")

if __name__ == "__main__":
    main()