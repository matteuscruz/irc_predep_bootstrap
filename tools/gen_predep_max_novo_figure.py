#!/usr/bin/env python3
"""
Invólucro de linha de comando para `src.visualization.predominancia_plots`.

A implementação vive no módulo, para a pipeline e a linha de comando gerarem a
mesma figura. Mantido porque os comandos documentados apontam para cá.

Uso:
  python tools/gen_predep_max_novo_figure.py [--nc ...] [--out plots/x.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.visualization.predominancia_plots import (  # noqa: E402
    DEFAULT_NC_PREDEP, plot_predep_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=DEFAULT_NC_PREDEP)
    ap.add_argument("--out", default="plots/exp22_predep_max_novo.png")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    plot_predep_max(args.nc, out.parent, tol=args.tol, filename=out.name,
                    dpi=args.dpi)


if __name__ == "__main__":
    main()
