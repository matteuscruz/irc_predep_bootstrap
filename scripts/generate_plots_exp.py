#!/usr/bin/env python3
"""
Gera todos os plots de um experimento PREDEP granular.

Para cada MoV em results/predep_granular_brazil/expN/:
  - Gera plots individuais de cada bacia encontrada
  - Gera plots do Brasil completo (arquivo merged)

Uso:
  python scripts/generate_plots_exp.py --exp-n 21
  python scripts/generate_plots_exp.py
  python scripts/generate_plots_exp.py --exp-n 21 --movs "ONI,NAO"
  python scripts/generate_plots_exp.py --exp-n 21 --lags "1,3,6,9,12"

Gerar PDFs por bacia (a partir de PNGs já gerados):
  python scripts/generate_plots_exp.py --exp-n 21 --pdf-basin parana
  python scripts/generate_plots_exp.py --exp-n 21 --pdf-basin parana --movs "ONI,AMO"
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization.cluster_plots import generate_plots_granular  # noqa: E402


def latest_exp(results_base: Path) -> int:
    nums = [
        int(m.group(1))
        for d in results_base.iterdir()
        if d.is_dir() and (m := re.match(r"^exp(\d+)$", d.name))
    ]
    if not nums:
        raise FileNotFoundError(
            f"Nenhum experimento encontrado em {results_base}"
        )
    return max(nums)


def _process_mov_dir(mov_dir: Path, plots_base: Path, lags: list[int] | None) -> int:
    mov = mov_dir.name
    plots_dir = plots_base / mov
    nc_files = sorted(mov_dir.glob("*.nc"))
    merged_name = f"{mov}_predep_granular_seasonal.nc"
    basin_ncs = [f for f in nc_files if f.name != merged_name]
    merged_nc = mov_dir / merged_name
    has_merged = merged_nc.exists()

    print(f"\n[{mov}]  {len(basin_ncs)} bacia(s)  merged={'sim' if has_merged else 'não'}")
    count = 0
    for nc in basin_ncs:
        basin_name = nc.stem.replace(f"{mov}_", "").replace("_predep_granular_seasonal", "")
        print(f"  -> bacia: {basin_name}")
        generate_plots_granular(nc, mov, plots_dir, basin_name=basin_name, lags=lags)
        count += 1

    if has_merged:
        print("  -> Brasil (merged)")
        generate_plots_granular(merged_nc, mov, plots_dir, basin_name="Brasil", lags=lags)
        count += 1

    return count


def generate_for_exp(
    exp_n: int,
    mov_filter: list[str] | None = None,
    lags: list[int] | None = None,
):
    results_base = ROOT / "results" / "predep_granular_brazil"
    exp_dir = results_base / f"exp{exp_n:02d}"

    if not exp_dir.exists():
        print(f"[ERRO] Experimento não encontrado: {exp_dir}")
        sys.exit(1)

    plots_base = ROOT / "plots" / f"exp{exp_n:02d}"
    mov_dirs = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir()
    )

    if mov_filter:
        mov_dirs = [d for d in mov_dirs if d.name in mov_filter]
        missing = set(mov_filter) - {d.name for d in mov_dirs}
        if missing:
            print(f"[AVISO] MoVs não encontrados no exp{exp_n}: {sorted(missing)}")

    lags_label = f"lags={lags}" if lags else "todos os lags"
    print(f"Experimento : exp{exp_n}  ({lags_label})")
    print(f"MoVs        : {[d.name for d in mov_dirs]}")
    print(f"Plots em    : {plots_base}/")
    print("=" * 60)

    total_saved = sum(_process_mov_dir(d, plots_base, lags) for d in mov_dirs)

    print(f"\n{'=' * 60}")
    print(f"Concluido — {total_saved} conjunto(s) de plots em {plots_base}/")


def _make_pdf(png_paths: list[Path], output_path: Path) -> None:
    """Empacota uma lista de PNGs em um único PDF (uma página por PNG) via Pillow."""
    imgs = []
    for p in png_paths:
        img = Image.open(str(p)).convert("RGB")
        imgs.append(img)
    if not imgs:
        return
    first, rest = imgs[0], imgs[1:]
    first.save(str(output_path), save_all=True, append_images=rest)
    for img in imgs:
        img.close()
    print(f"  PDF: {output_path.name}  ({len(png_paths)} páginas)")


def _collect_basin_pngs(plots_base: Path, mov_dirs: list[Path], basin_name: str):
    """
    Para cada MoV retorna listas de PNGs agrupadas por categoria.
    Ordem dentro de cada MoV: optimal primeiro, lags depois.
    """
    predep: list[Path] = []
    regr: list[Path] = []
    comp: list[Path] = []

    for mov_dir in mov_dirs:
        mov = mov_dir.name
        predep_dir = plots_base / mov / "PREDEP"   / "BACIAS" / basin_name
        regr_dir   = plots_base / mov / "REGRESSAO" / "BACIAS" / basin_name
        comp_dir   = plots_base / mov / "COMPARACAO" / "BACIAS" / basin_name

        for src_dir, collection, patterns in [
            (predep_dir, predep, ["granular_optimal_*.png", "granular_*lags_*.png"]),
            (regr_dir,   regr,   ["granular_optimal_*.png", "granular_*lags_*.png"]),
            (comp_dir,   comp,   ["winner_optimal_*.png",   "diff_*lags_*.png"]),
        ]:
            if not src_dir.exists():
                continue
            for pattern in patterns:
                candidates = sorted(src_dir.glob(pattern))
                if candidates:
                    collection.append(candidates[-1])

    return predep, regr, comp


def generate_pdfs_for_basin(
    exp_n: int,
    basin_name: str,
    mov_filter: list[str] | None = None,
):
    """
    Para uma bacia, coleta todos os PNGs já gerados de cada MoV e gera:
      plots/PDF/{basin_name}_PREDEP.pdf      (optimal + lags por MoV)
      plots/PDF/{basin_name}_REGRESSAO.pdf   (optimal + lags por MoV)
      plots/PDF/{basin_name}_COMPARACAO.pdf  (winner + diff por MoV)
    """
    results_base = ROOT / "results" / "predep_granular_brazil"
    exp_dir = results_base / f"exp{exp_n:02d}"

    if not exp_dir.exists():
        print(f"[ERRO] Experimento não encontrado: {exp_dir}")
        sys.exit(1)

    plots_base = ROOT / "plots" / f"exp{exp_n:02d}"
    mov_dirs = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir()
    )
    if mov_filter:
        mov_dirs = [d for d in mov_dirs if d.name in mov_filter]

    predep_pngs, regr_pngs, comp_pngs = _collect_basin_pngs(plots_base, mov_dirs, basin_name)

    pdf_dir = plots_base / "PDF"
    pdf_dir.mkdir(exist_ok=True)

    n_movs = len(predep_pngs) // 2 if predep_pngs else 0
    print(f"\nBacia       : {basin_name}")
    print(f"MoVs        : {n_movs}")
    print(f"PDFs em     : {pdf_dir}/")
    print("=" * 60)

    for pngs, label in [
        (predep_pngs, "PREDEP"),
        (regr_pngs,   "REGRESSAO"),
        (comp_pngs,   "COMPARACAO"),
    ]:
        if pngs:
            _make_pdf(pngs, pdf_dir / f"{basin_name}_{label}.pdf")
        else:
            print(f"  [AVISO] Nenhum PNG {label} encontrado para bacia '{basin_name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-n", type=int, default=0,
        help="Número do experimento (padrão: mais recente)",
    )
    parser.add_argument(
        "--movs", type=str, default="",
        help="MoVs separados por vírgula (padrão: todos)",
    )
    parser.add_argument(
        "--lags", type=str, default="0,1,3,6,9,12",
        help=(
            "Lags a exibir, separados por vírgula (padrão: 0,1,3,6,9,12). "
            "Use 'all' para todos os lags disponíveis. Ex: --lags '1,3,6,9,12'"
        ),
    )
    parser.add_argument(
        "--pdf-basin", type=str, default="",
        help=(
            "Gera PDFs por bacia a partir dos PNGs já existentes. "
            "Ex: --pdf-basin parana"
        ),
    )
    args = parser.parse_args()

    results_base = ROOT / "results" / "predep_granular_brazil"
    exp_n = args.exp_n if args.exp_n > 0 else latest_exp(results_base)
    mov_filter = [m.strip() for m in args.movs.split(",") if m.strip()] or None

    if args.pdf_basin:
        generate_pdfs_for_basin(exp_n, args.pdf_basin.strip(), mov_filter)
        return

    lags = (
        None if args.lags.strip().lower() == "all"
        else [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    )
    generate_for_exp(exp_n, mov_filter, lags)


if __name__ == "__main__":
    main()
