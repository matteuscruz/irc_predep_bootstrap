#!/usr/bin/env python3
"""
Invólucro de linha de comando para `src.visualization.predominancia_plots`.

A implementação vive no módulo, para a pipeline e a linha de comando gerarem a
mesma figura. Mantido porque os comandos documentados apontam para cá.

Uso:
  python tools/gen_som_completo_figure.py [--k 4] [--out plots/x.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.visualization.predominancia_plots import (  # noqa: E402
    DEFAULT_NC_SOM, plot_som_regimes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=DEFAULT_NC_SOM)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--top-movs", type=int, default=10)
    ap.add_argument("--out", default="plots/exp26_som_completo.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    plot_som_regimes(args.nc, out.parent, k=args.k, top_movs=args.top_movs,
                     filename=out.name, dpi=args.dpi)


if __name__ == "__main__":
    main()
