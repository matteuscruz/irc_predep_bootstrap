#!/usr/bin/env python3
"""
Download dos resultados de regressão do volume Modal para disco local.

Uso:
    python scripts/download_regression_results.py           # todos os índices
    python scripts/download_regression_results.py TNA AMO   # índices específicos
"""

import sys
import subprocess
from pathlib import Path

import modal

VOLUME_NAME = "xavier-data"
REMOTE_BASE = "results/regression_brazil"
LOCAL_BASE = Path("results/regression")

volume = modal.Volume.from_name(VOLUME_NAME)


def list_remote_indices():
    """Lista os índices disponíveis no volume Modal."""
    entries = volume.listdir(REMOTE_BASE)
    return sorted(
        e.path.split("/")[-1]
        for e in entries
        if e.type.name == "DIRECTORY"
    )


def download_index(mov_name: str) -> bool:
    local_dir = LOCAL_BASE / mov_name
    local_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME,
         f"{REMOTE_BASE}/{mov_name}",
         str(local_dir)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  ERRO {mov_name}: {result.stderr.strip()}")
        return False

    files = list(local_dir.glob("*.nc"))
    print(f"  {mov_name}: {len(files)} arquivo(s) -> {local_dir}/")
    return True


def main():
    if len(sys.argv) > 1:
        indices = sys.argv[1:]
    else:
        print(f"Listando índices disponíveis em '{REMOTE_BASE}'...")
        indices = list_remote_indices()
        print(f"  {len(indices)} encontrados: {', '.join(indices)}\n")

    print(f"Baixando {len(indices)} índice(s) para {LOCAL_BASE}/\n")
    ok, fail = 0, 0
    for mov in indices:
        if download_index(mov):
            ok += 1
        else:
            fail += 1

    print(f"\nConcluído: {ok} ok, {fail} falha(s)")
    print(f"Resultados em: {LOCAL_BASE}/")


if __name__ == "__main__":
    main()
