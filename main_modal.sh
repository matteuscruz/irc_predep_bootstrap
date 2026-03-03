#!/bin/bash
set -e

echo "=========================================="
echo "🚀 PREDEP Modal - Setup e Execução"
echo "=========================================="

# Verificar se Modal está instalado
if ! command -v modal &> /dev/null; then
    echo "📦 Instalando Modal..."
    pip install modal
fi

# Verificar autenticação
echo "🔐 Verificando autenticação Modal..."
if [ ! -f "$HOME/.modal.toml" ]; then
    echo "⚠️  Modal não autenticado. Execute:"
    echo "   modal token new"
    exit 1
fi
echo "   ✓ Autenticado"

# Criar volume
echo "📦 Criando volume Modal..."
modal volume create xavier-data 2>/dev/null || echo "   Volume já existe"

# Upload dados XAVIER
echo "📤 Upload dados XAVIER..."
if [ -f "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc" ]; then
    modal volume put xavier-data \
        data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc \
        /input/pr_monthly_1961_2019_BRAZIL_combined.nc --force 2>/dev/null || echo "   ✓ Dados XAVIER já existem"
elif [ -f "data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc" ]; then
    echo "   ⚠️  Usando dados do RS (Brasil completo não encontrado)"
    modal volume put xavier-data \
        data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc \
        /input/pr_monthly_1961_2019_BRAZIL_combined.nc --force 2>/dev/null || echo "   ✓ Dados já existem"
else
    echo "   ⚠️  Nenhum arquivo XAVIER encontrado"
    echo "   Procure por: pr_monthly_1961_2019_BRAZIL_combined.nc ou pr_monthly_1961_2019_RS_combined.nc"
    exit 1
fi

# Upload índice AAO
echo "📤 Upload índice AAO..."
if [ -f "data/raw/mov_indices/AAO_indice data.csv" ]; then
    modal volume put xavier-data \
        "data/raw/mov_indices/AAO_indice data.csv" \
        /input/AAO_indice_data.csv --force 2>/dev/null || echo "   ✓ Índice AAO já existe"
else
    echo "   ⚠️  Arquivo não encontrado: data/raw/mov_indices/AAO_indice data.csv"
    exit 1
fi

# Executar análise
echo ""
echo "=========================================="
echo "🔄 Executando análise no Modal..."
echo "=========================================="
modal run experiments/02_xavier_rs_analysis_MODAL.py ${1:+--n-boot $1}

# Download resultados
echo ""
echo "=========================================="
echo "💾 Baixando resultados..."
echo "=========================================="
mkdir -p results/modal_xavier_brazil
modal volume get xavier-data results/xavier_brazil results/modal_xavier_brazil/ 2>/dev/null || echo "   ⚠️  Nenhum resultado para baixar ainda"

echo ""
echo "=========================================="
echo "✅ CONCLUÍDO!"
echo "=========================================="
echo "📁 Resultados em: results/modal_xavier_rs/"
echo ""
echo "📊 Arquivos gerados:"
ls -lh results/modal_xavier_rs/*.nc 2>/dev/null || echo "   Nenhum arquivo .nc encontrado"
