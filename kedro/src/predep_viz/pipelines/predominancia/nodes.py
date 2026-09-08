"""
Metodologia da predominância — exp 22 → 23 → 26 → figuras.

O que esta pipeline reproduz: um (lag, MoV) por estação escolhido na bacia
inteira pela predominância com concordância entre Xavier e CHIRPS, os mapas de
α daí decorrentes, e o SOM por trimestre treinado sob a máscara de concordância.
Documentado em `METODOLOGIA_PREDOMINANCIA.md`.

**Modo resume por padrão**, igual ao `predep_production`. Os dois primeiros
passos rodam no Modal sobre as fatias em `/slices18` (4 estações × 7 lags × 7
deslocamentos, escritas pelo `experiments/18_modal_mapa_excesso_nulo.py`) e não
recalculam PREDEP — só reduzem o grid já calculado. Ainda assim exigem sessão
Modal autenticada, então com `allow_modal_dispatch: false` os nós apenas
verificam que os NetCDFs já baixados existem.

O passo do SOM é local e custa ~15 min (192 SOMs). Ele é pulado quando a saída
já existe, a menos que `forcar_som: true`.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

# kedro/src/predep_viz/pipelines/predominancia/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

VOLUME = "predep-common-window"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _rodar(cmd: list[str], descricao: str) -> None:
    print(f"    [+] {descricao}: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{descricao} falhou ({r.returncode}):\n{r.stderr[-2000:]}")
    if r.stdout:
        print("       " + r.stdout.strip().replace("\n", "\n       ")[-1500:])


def preparar_grids(params: dict, allow_modal_dispatch: bool) -> dict:
    """Garante os dois NetCDFs que vêm do Modal: exp22 (α) e exp23 (features).

    Em modo resume só confere que existem localmente — é o caminho normal, já
    que ambos são redução determinística de `/slices18` e não mudam entre
    execuções. Com dispatch ligado, roda o `modal run` e baixa o resultado.
    """
    saidas = {}
    for chave, script, remoto in (
        ("predep", "experiments/22_modal_predep_max_novo.py",
         "/results/exp22_predep_max_novo.nc"),
        ("features", "experiments/23_modal_export_som_features.py",
         "/results/exp23_som_features.nc"),
    ):
        local = ROOT / params[chave]
        if allow_modal_dispatch:
            _rodar(["modal", "run", script], f"exp {chave} no Modal")
            local.unlink(missing_ok=True)     # `modal volume get` recusa sobrescrever
            _rodar(["modal", "volume", "get", VOLUME, remoto, str(local)],
                   f"baixando {chave}")
        if not local.exists():
            raise FileNotFoundError(
                f"{local} não existe. Rode com "
                f"`--params predominancia.allow_modal_dispatch=true` para gerar no "
                f"Modal, ou baixe com:\n"
                f"  modal volume get {VOLUME} {remoto} {params[chave]}")
        saidas[chave] = str(local)
        print(f"    [+] {chave}: {local.name} ({local.stat().st_size/1e6:.1f} MB)")
    return saidas


def rodar_som(grids: dict, params: dict, forcar: bool) -> str:
    """Treina os 8 SOMs (4 trimestres × 2 fontes) e guarda regimes e assinaturas.

    Pula se a saída já existe: são ~15 min e o resultado é determinístico
    (sementes fixas em `SEEDS_TETO`), então repetir não acrescenta nada — é
    justamente essa determinação que o nó de reprodutibilidade confere.
    """
    saida = ROOT / params["som"]
    if saida.exists() and not forcar:
        print(f"    [=] {saida.name} já existe — pulando (use forcar_som: true)")
        return str(saida)
    _rodar([sys.executable, "experiments/26_som_completo_assinaturas.py",
            "--nc", params["features"], "--out", params["som"],
            "--n-regimes", *[str(k) for k in params["n_regimes"]],
            "--tol", str(params["tol"])],
           "SOM por trimestre (exp26)")
    return str(saida)


def gerar_figuras(grids: dict, som_nc: str, plots_dir: str, k: int) -> pd.DataFrame:
    """As duas figuras finais, pelo módulo em `src/visualization`.

    O sha256 vai na tabela porque é o que torna a reprodutibilidade
    verificável: duas execuções da pipeline têm de produzir o mesmo hash.
    """
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization.predominancia_plots import (plot_predep_max,
                                                       plot_som_regimes)

    caminhos = [
        plot_predep_max(grids["predep"], ROOT / plots_dir),
        plot_som_regimes(som_nc, ROOT / plots_dir, k=k),
    ]
    return pd.DataFrame([{"figura": p.name,
                          "bytes": p.stat().st_size,
                          "sha256": _sha256(p)} for p in caminhos])


def resumir_metricas(grids: dict, som_nc: str, k: int) -> pd.DataFrame:
    """Os números que uma reexecução tem de reproduzir, um por linha.

    É o artefato de reprodutibilidade: em vez de comparar figuras no olho,
    compara-se este CSV entre execuções.
    """
    import numpy as np
    import xarray as xr

    dp = xr.open_dataset(grids["predep"])
    dsom = xr.open_dataset(som_nc)
    movs = [str(m) for m in dp.mov_name.values]
    lags = dp.lag.values
    seasons = [str(s) for s in dp.season.values]
    ki = [int(v) for v in dsom.k.values].index(k)

    linhas = []
    for si, sn in enumerate(seasons):
        xav = dp["alpha"].values[0, 0, si]
        chi = dp["alpha"].values[1, 0, si]
        linhas.append({
            "estacao": sn,
            "mov": movs[int(dp["sel_mov"].values[0, si])],
            "lag": int(lags[int(dp["sel_lag"].values[0, si])]),
            "alpha_xavier": round(float(np.nanmean(xav)), 6),
            "alpha_chirps": round(float(np.nanmean(chi)), 6),
            "consenso": round(float(dp["consenso"].values[0, si]), 6),
            "consenso_surrogate_max": round(
                float(np.nanmax(dp["consenso"].values[1:, si])), 6),
            "ari_mascara": round(float(dsom["ari_mascara"].values[ki, si]), 6),
            "ari_nulo_max": round(float(np.nanmax(dsom["ari_nulo"].values[ki, si])), 6),
        })
    a_x = dp["alpha_ano"].values[0, 0]
    a_c = dp["alpha_ano"].values[1, 0]
    linhas.append({
        "estacao": "ANO",
        "mov": movs[int(dp["sel_mov_ano"].values[0])],
        "lag": int(lags[int(dp["sel_lag_ano"].values[0])]),
        "alpha_xavier": round(float(np.nanmean(a_x)), 6),
        "alpha_chirps": round(float(np.nanmean(a_c)), 6),
        "consenso": round(float(dp["consenso_ano"].values[0]), 6),
        "consenso_surrogate_max": round(float(np.nanmax(dp["consenso_ano"].values[1:])), 6),
        "ari_mascara": float("nan"),
        "ari_nulo_max": float("nan"),
    })
    df = pd.DataFrame(linhas)
    print("\n" + df.to_string(index=False))
    return df
