#!/bin/bash

echo "Executando testes unitários..."

# Executar testes básicos
echo "=== Testes básicos ==="
uv run pytest tests/ -v

# Executar testes com cobertura
echo "=== Testes com cobertura ==="
uv run pytest tests/ --cov=src --cov-report=term-missing

# Executar apenas testes rápidos
echo "=== Testes rápidos ==="
uv run pytest tests/ -m "not slow" -v

echo "Testes concluídos!"