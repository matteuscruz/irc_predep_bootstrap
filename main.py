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
import os
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
                                 VOLUME_MODAL, estimar_custo, fator_nboot, tag_janela)

LIMITE_AVISO_H = 2.0          # acima disto, o modo local confirma antes de começar


def baixar(remoto, local):
    """`modal volume get` para um arquivo temporário, que só substitui o local no fim.

    A versão anterior apagava o arquivo local ANTES de baixar: se a conta estivesse
    desativada ou o arquivo não existisse no volume, a cópia boa se perdia.
    """
    destino = ROOT / local
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    parcial.unlink(missing_ok=True)            # `volume get` recusa sobrescrever
    r = subprocess.run(["modal", "volume", "get", VOLUME_MODAL, remoto,
                        str(parcial)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0 and parcial.exists() and parcial.stat().st_size > 0:
        os.replace(parcial, destino)
        print(f"  ✅ {local}")
        return True
    parcial.unlink(missing_ok=True)
    # O Modal desenha o erro numa caixa; a linha útil é a que tem texto além da moldura.
    linhas = [ln.strip("│╭╮╰╯─ ").strip() for ln in (r.stderr or "").splitlines()]
    linhas = [ln for ln in linhas if ln and ln.lower() != "error"]
    motivo = linhas[-1] if linhas else "arquivo vazio"
    print(f"  ⚠ {remoto}: {motivo}"
          + (" — a cópia local anterior foi mantida" if destino.exists() else ""))
    return False


ARTEFATOS = "artifacts"


def artefato(tag, *partes):
    """Caminho local de um artefato: `artifacts/<config>/<partes...>`.

    `<config>` é o namespace da corrida (janela, anomalias, n_boot): resultados de
    configurações diferentes nunca se sobrescrevem. Dentro dele:

        raw/basins/<bacia>.nc                 α, r², escolha, nulo e SOM da bacia
        plots/basins/predep/<bacia>.png       figura do PREDEP
        plots/basins/regression/<bacia>.png   figura da regressão (r²)
        som/<mapeamento>.nc                   mapeamento espacial de um SOM
        som/plots/basins/<bacia>.png          SOM treinado na própria bacia
    """
    return "/".join([ARTEFATOS, tag, *partes])


def baixar_bacias(bacias, tag, com_figuras=True):
    """Traz de cada bacia o NetCDF, as figuras e o SOM para `artifacts/<config>/`."""
    print("\nbaixando os resultados...")
    for b in bacias:
        baixar(f"/results/{tag}/bacias/{b}.nc", artefato(tag, "raw", "basins", f"{b}.nc"))
        baixar(f"/results/{tag}/som_bacias/{b}.nc", artefato(tag, "som", f"{b}.nc"))
        if com_figuras:
            baixar(f"/results/{tag}/figuras/{b}_predep.png",
                   artefato(tag, "plots", "basins", "predep", f"{b}.png"))
            baixar(f"/results/{tag}/figuras/{b}_regressao.png",
                   artefato(tag, "plots", "basins", "regression", f"{b}.png"))
            baixar(f"/results/{tag}/figuras/{b}_som.png",
                   artefato(tag, "som", "plots", "basins", f"{b}.png"))


def nome_unidade(regiao, bacias):
    """Nome da saída de --som/--regiao. MESMA regra do entrypoint em run_bacias.py."""
    if regiao:
        return regiao
    return "brasil" if sorted(bacias) == sorted(resolver_bacias("all")) else "_".join(bacias)


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
           "--janela", args.janela,
           "--n-boot", str(args.n_boot)]
    if args.so_figuras:
        cmd.append("--so-figuras")
    if args.anomalias:
        cmd.append("--anomalias")
    if args.juntar:
        cmd.append("--juntar")
    if args.regiao:
        cmd += ["--regiao", args.regiao]
    if args.som:
        cmd.append("--som")
        cmd += ["--metrica", args.metrica]
    if args.busca_som:
        cmd.append("--busca-som")
    if args.busca_par_fixo:
        cmd.append("--busca-par-fixo")
    if args.busca_todos:
        cmd.append("--busca-todos")
    if args.usar_fatias_existentes:
        cmd.append("--usar-fatias-existentes")
    print("\n  " + " ".join(cmd) + "\n")
    rc = 1
    try:
        rc = subprocess.run(cmd, cwd=ROOT).returncode
    finally:
        # SEMPRE baixa, mesmo se a corrida falhar ou for interrompida: o que ficou
        # pronto no volume antes da falha (a conta cair, o timeout de uma figura)
        # vem para o repositório. `baixar` não apaga a cópia local se falhar.
        if not args.sem_baixar:
            if rc != 0:
                print(f"\n  ⚠ a corrida não terminou (código {rc}); baixando o que "
                      "ficou pronto no volume")
            baixar_resultados(args, bacias)
    return rc


def baixar_resultados(args, bacias):
    """Traz para o repositório o que a corrida produz, conforme o modo."""
    tag = tag_janela(args.janela, args.anomalias, args.n_boot)
    if args.juntar:
        baixar(f"/results/{tag}/brasil.nc", artefato(tag, "raw", "basins", "brasil.nc"))
    elif args.busca_som:
        # MESMO nome do entrypoint em run_bacias.py
        nome = "brasil_par_fixo" if args.busca_par_fixo else "brasil"
        baixar(f"/results/{tag}/som_busca/{nome}.nc",
               artefato(tag, "som", "busca", f"{nome}.nc"))
        baixar(f"/results/{tag}/figuras_som_busca/{nome}_som_busca.png",
               artefato(tag, "som", "plots", "busca", f"{nome}.png"))
    elif args.som:
        nome = nome_unidade(args.regiao, bacias)
        if args.metrica != "alpha":          # MESMO nome de som_regiao
            nome = f"{nome}_{args.metrica}"
        baixar(f"/results/{tag}/som/{nome}.nc", artefato(tag, "som", f"{nome}.nc"))
        baixar(f"/results/{tag}/figuras_som/{nome}_som.png",
               artefato(tag, "som", "plots", f"{nome}.png"))
    elif args.regiao:
        r = args.regiao
        baixar(f"/results/{tag}/regioes/{r}.nc", artefato(tag, "raw", "regions", f"{r}.nc"))
        baixar(f"/results/{tag}/som_regioes/{r}.nc", artefato(tag, "som", f"{r}.nc"))
        baixar(f"/results/{tag}/figuras_regioes/{r}_predep.png",
               artefato(tag, "plots", "regions", "predep", f"{r}.png"))
        baixar(f"/results/{tag}/figuras_regioes/{r}_regressao.png",
               artefato(tag, "plots", "regions", "regression", f"{r}.png"))
        baixar(f"/results/{tag}/figuras_regioes/{r}_som.png",
               artefato(tag, "som", "plots", "regions", f"{r}.png"))
    else:
        baixar_bacias(bacias, tag)


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
    n_sh = len([x for x in args.shifts.split(",") if x.strip() != ""])
    print(f"janela comum: {ds.sizes['time']} meses · {len(lat):,} pixels · "
          f"{len(movs)} MoVs")
    print(f"pedido: {n_mov} MoVs × {n_lag} lags × 4 estações × 2 fontes × "
          f"{n_sh} deslocamentos · n_boot {args.n_boot or 'padrão'} "
          f"(custo ×{fator_nboot(args.n_boot):.1f})\n")
    print(f"{'bacia':<30}{'pixels':>9}{'núcleo-hora':>14}")
    print("-" * 53)
    linhas = [(b, len(pixels_da_bacia(b, lat, lon)))
              for b in resolver_bacias(args.bacia)]
    linhas.sort(key=lambda t: t[1])
    total_px = total_h = 0
    for b, n in linhas:
        h = estimar_custo(n, n_mov, n_lag, n_proc=1, n_boot=args.n_boot) * n_sh / 3600
        total_px += n
        total_h += h
        print(f"{b:<30}{n:>9,}{h:>14,.0f}")
    print("-" * 53)
    print(f"{'TOTAL':<30}{total_px:>9,}{total_h:>14,.0f}")
    print("\n(núcleo-hora de cálculo das fatias, a partir de 13,4 ms por chamada de "
          "predep medidos localmente; o Modal cobra também memória e tempo ocioso)")


def assinaturas(args):
    """Tabela de quais MoVs distinguem cada regime, por bacia."""
    from src.analysis.som_regimes import tabela_assinaturas

    for b in resolver_bacias(args.bacia):
        nc = ROOT / f"{args.out_dir}/{tag_janela(args.janela, args.anomalias, args.n_boot)}/{b}.nc"
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
    ap.add_argument("--n-boot", type=int, default=0,
                    help="reamostragens por estimativa de α; 0 = ⌈n ln n⌉ do "
                         "Algoritmo 1 (627 com n=129). Outro valor vai para um "
                         "namespace próprio (sufixo _nb<valor>), sem misturar fatias")
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
    ap.add_argument("--regiao", default="",
                    help="nome da região (ex.: sul_sudeste, brasil): trata as bacias de "
                         "--bacia como uma unidade só e refaz escolha, surrogates e SOM "
                         "a partir das fatias, sem recalcular PREDEP")
    ap.add_argument("--som", action="store_true",
                    help="só calcula o SOM das bacias de --bacia a partir dos <bacia>.nc "
                         "já prontos (sem fatias, PREDEP ou surrogates); nome da saída "
                         "= --regiao, ou 'brasil' para todas, ou as bacias unidas por _")
    ap.add_argument("--metrica", choices=["alpha", "r2"], default="alpha",
                    help="com --som: agrupa pelo PREDEP (alpha) ou pela regressão (r2); "
                         "a máscara usa a mesma tolerância relativa nas duas")
    ap.add_argument("--busca-som", action="store_true",
                    help="busca o SOM mais concordante entre Xavier e CHIRPS (grade × K) "
                         "sobre o BRASIL INTEIRO numa passada, e roda a MESMA busca em "
                         "cada surrogate; exige as 12 bacias prontas")
    ap.add_argument("--busca-par-fixo", action="store_true",
                    help="com --busca-som: os surrogates usam o par (lag, MoV) do "
                         "observado em vez de refazer a escolha — 288 fatias em vez "
                         "de 2.016, mas o nulo deixa de incluir a seleção do par")
    ap.add_argument("--busca-todos", action="store_true",
                    help="com --busca-som: roda os 6 surrogates em todas as estações. "
                         "Sem isto, cada estação para no 1º surrogate que empatar ou "
                         "superar o observado — o veredito é o mesmo, com menos custo")
    ap.add_argument("--usar-fatias-existentes", action="store_true",
                    help="não calcula fatias: monta cada bacia só com as que já estão "
                         "no volume. Combinações (estação, lag) com alguma fatia "
                         "ausente saem da escolha de TODAS as séries, observado "
                         "inclusive, para o teste contra o nulo continuar justo")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o que seria feito, sem calcular")
    ap.add_argument("--sim", action="store_true",
                    help="não pergunta antes de corridas longas")
    args = ap.parse_args()

    if args.listar:
        return listar(args)

    if args.assinaturas:
        return assinaturas(args)

    if (args.regiao or args.som) and args.juntar:
        raise SystemExit("--juntar não combina com --regiao nem com --som")
    if (args.regiao or args.som) and args.executor != "modal":
        raise SystemExit("--regiao e --som só rodam no Modal")
    if args.som and args.so_figuras:
        raise SystemExit("--som já gera a figura do SOM; não use com --so-figuras")
    if args.busca_som and (args.som or args.regiao or args.juntar or args.so_figuras):
        raise SystemExit("--busca-som não combina com --som, --regiao, --juntar "
                         "nem --so-figuras")
    if args.busca_som and args.executor != "modal":
        raise SystemExit("--busca-som só roda no Modal")
    if args.metrica != "alpha" and not args.som:
        raise SystemExit("--metrica só tem efeito com --som")
    if (args.busca_par_fixo or args.busca_todos) and not args.busca_som:
        raise SystemExit("--busca-par-fixo e --busca-todos só têm efeito com --busca-som")
    if args.busca_som and sorted(resolver_bacias(args.bacia)) != sorted(resolver_bacias("all")):
        raise SystemExit("--busca-som roda sempre sobre o Brasil inteiro: use --bacia all")

    if args.juntar and args.executor == "local":
        from src.pipeline.juntar import juntar
        pasta = ROOT / args.out_dir / tag_janela(args.janela, args.anomalias, args.n_boot)
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
        if args.usar_fatias_existentes:
            print("\n  modo: monta só com as fatias já calculadas; combinações "
                  "incompletas saem da escolha de todas as séries")
        if args.juntar:
            print("\n  modo: juntar as bacias prontas num brasil.nc (nada é recalculado)")
        if args.busca_som:
            n_sur = len([x for x in args.shifts.split(",") if x.strip() not in ("", "0")])
            n_fat = len(bacias) * 4 * n_sur * (1 if args.busca_par_fixo else len(lags))
            print("\n  modo: busca do SOM concordante (grade × K) sobre o Brasil inteiro, "
                  "no observado e nos surrogates")
            print("  nulo: " + ("surrogates com o par do observado" if args.busca_par_fixo
                                else "cada surrogate refaz a escolha do par sobre o Brasil"))
            print(f"  custo: até {n_fat:,} fatias de surrogate (as já existentes no volume "
                  f"são puladas) + 12 SOMs em cada uma das até {4 * (n_sur + 1)} "
                  "combinações estação × deslocamento")
            print("  parada antecipada: " + ("desligada (--busca-todos)" if args.busca_todos
                  else "cada estação para no 1º surrogate que empatar ou superar o "
                       "observado"))
        if args.som:
            print(f"\n  modo: só o SOM de '{nome_unidade(args.regiao, bacias)}' a partir "
                  "das bacias já calculadas (sem fatias, PREDEP ou surrogates)")
        elif args.regiao:
            print(f"\n  modo: região '{args.regiao}' — escolha, surrogates e SOM sobre a "
                  "união das bacias (PREDEP não é recalculado)")
        tag = tag_janela(args.janela, args.anomalias, args.n_boot)
        print(f"\n  janela: {args.janela} · "
              f"{'anomalias' if args.anomalias else 'totais brutos'}")
        if args.juntar:
            print(f"  saída:  {artefato(tag, 'raw', 'basins', 'brasil.nc')}")
        elif args.busca_som:
            nome = "brasil_par_fixo" if args.busca_par_fixo else "brasil"
            print(f"  saída:  {artefato(tag, 'som', 'busca', nome + '.nc')}  ·  "
                  f"{artefato(tag, 'som', 'plots', 'busca', nome + '.png')}")
        elif args.som:
            nome = nome_unidade(args.regiao, bacias)
            if args.metrica != "alpha":
                nome = f"{nome}_{args.metrica}"
            print(f"  métrica do SOM: {args.metrica}")
            print(f"  saída:  {artefato(tag, 'som', nome + '.nc')}  ·  "
                  f"{artefato(tag, 'som', 'plots', nome + '.png')}")
        elif args.regiao:
            print(f"  saída:  {artefato(tag, 'raw', 'regions', args.regiao + '.nc')}  ·  "
                  f"{artefato(tag, 'plots', 'regions')}/{{predep,regression}}/"
                  f"{args.regiao}.png")
        else:
            print(f"  saída:  {artefato(tag)}/")
            print("            raw/basins/<bacia>.nc · plots/basins/{predep,regression}/"
                  "<bacia>.png · som/<bacia>.nc · som/plots/basins/<bacia>.png")
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
