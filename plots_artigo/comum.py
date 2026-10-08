"""Caminhos e utilitários compartilhados pelos scripts de figura do artigo.

Todos os scripts só LEEM resultados prontos (.nc) — nenhum recalcula PREDEP ou SOM.
Os caminhos podem ser trocados por variáveis de ambiente:

    DADOS   pasta com os .nc do artigo         (padrão: plots_artigo/dados)
    SAIDA   pasta onde as figuras são gravadas (padrão: plots_artigo/figuras)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

DADOS = Path(os.environ.get("DADOS", Path(__file__).resolve().parent / "dados"))
SAIDA = Path(os.environ.get("SAIDA", Path(__file__).resolve().parent / "figuras"))
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def exigir(caminho: Path) -> Path:
    if not caminho.exists():
        raise SystemExit(f"arquivo de entrada não encontrado: {caminho}\n"
                         "veja plots_artigo/README.md, seção 'Dados de entrada'")
    return caminho
