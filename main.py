#!/usr/bin/env python3
"""
PREDEP — ponto de entrada único.

    python main.py --bacia uruguai --movs all
    python main.py --bacia all --movs all
    python main.py --bacia parana,uruguai --movs MEI,TSA,NIN04

Para cada bacia roda a cadeia inteira e grava um NetCDF em
`results/bacias/<bacia>.nc`:

  1. α (PREDEP) e r² por pixel, para cada estação × lag × MoV e as duas fontes
     de precipitação, com as MESMAS amostras e a MESMA semente nas duas.
  2. A escolha por predominância com concordância: UM (lag, MoV) por estação
     para a bacia inteira.
  3. O SOM por trimestre e por fonte, treinado sob a máscara de concordância e
     projetado em todos os pixels.

A metodologia e o porquê de cada decisão estão em `METODOLOGIA_PREDOMINANCIA.md`.
Toda a lógica vive em `src/` — este arquivo só resolve argumentos e chama.

**Roda no Modal.** É o padrão e deve continuar sendo: local, a mesma conta leva
~70 min para o Uruguai e mais de 50 h para as doze bacias, porque são
`pixels × MoVs × lags × estações × 2` chamadas de `predep` a ~13 ms cada. No
Modal cada (bacia, estação, lag) vai para o seu container.

`--executor local` existe só para depurar uma bacia pequena, e avisa antes de
começar. Não é o caminho de produção.

Subcomandos úteis antes de rodar:

    python main.py --listar             # bacias, pixels e custo estimado
    python main.py --bacia X --dry-run  # o que seria feito, sem calcular
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Só imports leves no topo. `src.pipeline.bacia` puxa scipy pela cadeia
# (grid_predep -> predep_calculation.core), e o despacho ao Modal tem de
# funcionar mesmo num ambiente onde essa pilha está quebrada — foi exatamente
# assim que este arquivo falhou antes de chegar ao `modal run`.
from src.data_processing.bacias import (resolver_bacias,  # noqa: E402
                                        resolver_movs)
from src.pipeline.config import (APP_MODAL, JANELA_COMUM,  # noqa: E402
                                 JANELA_MODAL_PADRAO, SAIDA_PADRAO,
                                 VOLUME_MODAL, estimar_custo, tag_janela)

LIMITE_AVISO_H = 2.0          # acima disto, o modo local confirma antes de começar


def baixar(remoto, local):
    """`modal volume get`, que recusa sobrescrever — por isso apaga antes."""
    destino = ROOT / local
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.unlink(missing_ok=True)
    r = subprocess.run(["modal", "volume", "get", VOLUME_MODAL, remoto,
                        str(destino)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✅ {local}")
        return True
    print(f"  ⚠ {remoto}: {r.stderr.strip().splitlines()[-1] if r.stderr else 'falhou'}")
    return False


def baixar_bacias(bacias, tag, com_figuras=True):
    """Traz o NetCDF e os PNGs de cada bacia do volume para o repo, sob o namespace
    da janela — para resultados de janelas diferentes não se sobrescreverem."""
    print("\nbaixando os resultados...")
    for b in bacias:
        baixar(f"/results/{tag}/bacias/{b}.nc", f"{SAIDA_PADRAO}/{tag}/{b}.nc")
        if com_figuras:
            for suf in ("predep", "som"):
                baixar(f"/results/{tag}/figuras/{b}_{suf}.png",
                       f"plots/bacias/{tag}/{b}_{suf}.png")


def despachar_modal(args, bacias):
    """Chama `modal run` — a resolução de MoVs acontece lá dentro.

    O entrypoint local do app não importa numpy nem xarray, e aqui também não
    passamos por eles: o ambiente local tem numpy 2.x incompatível com o xarray
    instalado, e o despacho não pode depender disso para funcionar.
    """
    lags = ",".join(str(x) for x in range(args.lag_min, args.lag_max + 1))
    cmd = ["modal", "run", APP_MODAL,
           "--bacias", ",".join(bacias),
           "--movs", args.movs,
           "--lags", lags,
           "--shifts", args.shifts,
           "--n-regimes", str(args.n_regimes),
           "--tol", str(args.tol),
           "--iters", str(args.iters),
           "--seed", str(args.seed),
           "--janela", args.janela]
    if args.so_figuras:
        cmd.append("--so-figuras")
    if args.anomalias:
        cmd.append("--anomalias")
    if args.juntar:
        cmd.append("--juntar")
    print("\n  " + " ".join(cmd) + "\n")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc == 0 and not args.sem_baixar:
        tag = tag_janela(args.janela, args.anomalias)
        if args.juntar:
            baixar(f"/results/{tag}/brasil.nc", f"{SAIDA_PADRAO}/{tag}/brasil.nc")
        else:
            baixar_bacias(bacias, tag)
    return rc


def _abrir_janela(caminho):
    """Abre a janela comum. Só o modo local e o `--listar` precisam disto."""
    try:
        import xarray as xr
    except Exception as e:                       # pilha científica quebrada
        raise SystemExit(
            f"não consegui importar xarray: {type(e).__name__}: {e}\n"
            "Isto só é necessário para --listar e para --executor local.\n"
            "O despacho ao Modal não precisa: use\n"
            "    python main.py --bacia <bacia> --movs all")
    p = Path(caminho)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(
            f"dataset da janela comum não encontrado: {p}\n"
            "Ele é construído por `python -m src.data_processing.janela_comum` "
            "e reúne Xavier + CHIRPS + 29 MoVs em 516 meses (1982–2024).")
    return xr.open_dataset(p)


def listar(args):
    from src.data_processing.bacias import pixels_da_bacia

    ds = _abrir_janela(args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    movs = [str(m) for m in ds.mov_name.values]
    n_mov = len(resolver_movs(args.movs, movs))
    n_lag = len(range(args.lag_min, args.lag_max + 1))
    print(f"janela comum: {ds.sizes['time']} meses · {len(lat):,} pixels · "
          f"{len(movs)} MoVs")
    print(f"pedido: {n_mov} MoVs × {n_lag} lags × 4 estações × 2 fontes\n")
    print(f"{'bacia':<30}{'pixels':>9}{'custo estimado':>18}")
    print("-" * 57)
    linhas = [(b, len(pixels_da_bacia(b, lat, lon)))
              for b in resolver_bacias("all")]
    linhas.sort(key=lambda t: t[1])
    total_px = total_s = 0
    for b, n in linhas:
        s = estimar_custo(n, n_mov, n_lag, n_proc=args.n_proc or 8)
        total_px += n
        total_s += s
        print(f"{b:<30}{n:>9,}{s/60:>15.0f} min")
    print("-" * 57)
    print(f"{'TOTAL':<30}{total_px:>9,}{total_s/3600:>15.1f} h")
    print(f"\n(estimativa com {args.n_proc or 8} núcleos; medido em 13,4 ms por "
          "chamada de predep)")


def assinaturas(args):
    """Tabela de quais MoVs distinguem cada regime, por bacia."""
    from src.analysis.som_regimes import tabela_assinaturas

    for b in resolver_bacias(args.bacia):
        nc = ROOT / f"{args.out_dir}/{tag_janela(args.janela, args.anomalias)}/{b}.nc"
        if not nc.exists():                    # resultados anteriores ao namespace
            nc = ROOT / f"{args.out_dir}/{b}.nc"
        if not nc.exists():
            print(f"{b}: {nc} não existe — rode o cálculo primeiro")
            continue
        df = tabela_assinaturas(nc)
        print(f"\n=== {b} ===")
        print(df.to_string(index=False))
        destino = nc.with_name(f"{b}_assinaturas.csv")
        df.to_csv(destino, index=False)
        print(f"\n  ✅ {destino}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bacia", default="all",
                    help="nome, lista separada por vírgula, ou 'all' (padrão)")
    ap.add_argument("--movs", default="all",
                    help="nome, lista separada por vírgula, ou 'all' (padrão)")
    ap.add_argument("--lag-min", type=int, default=0)
    ap.add_argument("--lag-max", type=int, default=6)
    ap.add_argument("--shifts", default="0,3,5,7,9,11,13",
                    help="deslocamentos anuais do nulo; 0 = observado. "
                         "Sem eles α e consenso não têm contra o que ser lidos.")
    ap.add_argument("--janela", default=JANELA_MODAL_PADRAO,
                    help="arquivo da janela comum no volume Modal "
                         "(ex.: predep_janela_1982_2024.nc)")
    ap.add_argument("--anomalias", action="store_true",
                    help="remove a climatologia mensal da chuva antes do PREDEP")
    ap.add_argument("--n-regimes", type=int, default=4, help="regimes do SOM")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="|Δα| da máscara de concordância")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--iters", type=int, default=50, help="épocas do SOM")
    ap.add_argument("--executor", choices=["modal", "local"], default="modal",
                    help="onde calcular (padrão: modal)")
    ap.add_argument("--n-proc", type=int, default=None,
                    help="processos no modo local (padrão: todos os núcleos)")
    ap.add_argument("--nc", default=JANELA_COMUM)
    ap.add_argument("--out-dir", default=SAIDA_PADRAO)
    ap.add_argument("--pular-som", action="store_true")
    ap.add_argument("--so-figuras", action="store_true",
                    help="não recalcula: só redesenha as figuras da bacia")
    ap.add_argument("--sem-baixar", action="store_true",
                    help="deixa os resultados no volume, sem trazer para o repo")
    ap.add_argument("--pular-figuras", action="store_true",
                    help="no modo local, não desenha as figuras ao final")
    ap.add_argument("--sobrescrever", action="store_true",
                    help="refaz bacias que já têm .nc")
    ap.add_argument("--listar", action="store_true",
                    help="mostra bacias, pixels e custo estimado, e sai")
    ap.add_argument("--assinaturas", action="store_true",
                    help="imprime quais MoVs distinguem cada regime do SOM, e sai")
    ap.add_argument("--juntar", action="store_true",
                    help="não calcula: junta todas as bacias já prontas da janela "
                         "num brasil.nc e baixa")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o que seria feito, sem calcular")
    ap.add_argument("--sim", action="store_true",
                    help="não pergunta antes de corridas longas")
    args = ap.parse_args()

    if args.listar:
        return listar(args)

    if args.assinaturas:
        return assinaturas(args)

    if args.juntar and args.executor == "local":
        from src.pipeline.juntar import juntar
        pasta = ROOT / args.out_dir / tag_janela(args.janela, args.anomalias)
        return juntar(sorted(pasta.glob("*.nc")), pasta / "brasil.nc",
                      resolver_bacias("all"))

    # A resolução das bacias não precisa do xarray — é o diretório de
    # shapefiles. Só o modo local abre a janela comum aqui.
    bacias = resolver_bacias(args.bacia)
    lags = list(range(args.lag_min, args.lag_max + 1))

    print("=" * 62)
    n_sur = len([x for x in args.shifts.split(",") if x.strip() not in ("", "0")])
    print(f"PREDEP · {len(bacias)} bacia(s) · MoVs: {args.movs} · "
          f"{len(lags)} lags · {n_sur} surrogates · k={args.n_regimes} · "
          f"executor: {args.executor}")
    print("=" * 62)

    if args.executor == "modal":
        for b in bacias:
            print(f"  {b}")
        if args.so_figuras:
            print("\n  modo: só figuras (nada é recalculado)")
        if args.juntar:
            print("\n  modo: juntar as bacias prontas num brasil.nc (nada é recalculado)")
        tag = tag_janela(args.janela, args.anomalias)
        print(f"\n  janela: {args.janela} · "
              f"{'anomalias' if args.anomalias else 'totais brutos'}")
        if args.juntar:
            print(f"  saída:  {SAIDA_PADRAO}/{tag}/brasil.nc")
        else:
            print(f"  saída:  {SAIDA_PADRAO}/{tag}/<bacia>.nc  ·  plots/bacias/{tag}/<bacia>_*.png")
        if args.dry_run:
            print("\n--dry-run: nada foi despachado.")
            return
        return despachar_modal(args, bacias)

    from src.data_processing.bacias import pixels_da_bacia
    from src.pipeline.bacia import executar_varias

    ds = _abrir_janela(args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    movs_idx = resolver_movs(args.movs, [str(m) for m in ds.mov_name.values])
    tam = [(b, len(pixels_da_bacia(b, lat, lon))) for b in bacias]
    tam.sort(key=lambda t: t[1])
    horas = sum(estimar_custo(n, len(movs_idx), len(lags),
                              n_proc=args.n_proc or 8)
                for _, n in tam) / 3600
    for b, n in tam:
        print(f"  {b:<30}{n:>8,} px")
    print(f"\n  saída:  {args.out_dir}/<bacia>.nc")
    print(f"  custo estimado do PREDEP: {horas:.1f} h "
          f"({args.n_proc or 8} núcleos)")

    if args.dry_run:
        print("\n--dry-run: nada foi calculado.")
        return

    if horas > LIMITE_AVISO_H and not args.sim:
        print(f"\n⚠ Isso passa de {LIMITE_AVISO_H:.0f} h nesta máquina, e o "
              "padrão do projeto é o Modal.")
        print("    python main.py --bacia " + args.bacia + " --movs " + args.movs)
        r = input("\nRodar local mesmo assim? [s/N] ").strip().lower()
        if r not in ("s", "sim", "y", "yes"):
            print("cancelado.")
            return

    saidas = executar_varias(
        args.bacia, movs=args.movs, lags=lags, nc_janela=args.nc,
        out_dir=args.out_dir, n_regimes=args.n_regimes, tol=args.tol,
        seed=args.seed, n_proc=args.n_proc, iters=args.iters,
        pular_som=args.pular_som, sobrescrever=args.sobrescrever)

    if saidas and not args.pular_figuras:
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization.bacia_plots import gerar_figuras_bacia
        print("\ndesenhando as figuras...")
        for p in saidas:
            gerar_figuras_bacia(str(p), ROOT / "plots/bacias")

    print(f"\n{'=' * 62}")
    print(f"{len(saidas)} de {len(bacias)} bacia(s) concluída(s)")
    for p in saidas:
        print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:          # nome de bacia/MoV inválido: a mensagem
        raise SystemExit(f"\n{e}")  # já lista o que existe — traceback só atrapalha
    except KeyboardInterrupt:
        raise SystemExit("\ninterrompido.")
