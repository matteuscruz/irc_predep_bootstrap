"""
Janela comum Xavier ∩ CHIRPS ∩ MoVs — o dataset que a pipeline por bacia consome.

Substitui `notebooks/exploratory/common_window_dataset.ipynb` como produtor. O
notebook fixava os caminhos dos mensais e herdava a janela da interseção das
fontes; aqui início e fim são parâmetros, e o arquivo sai com o período no nome.
Uma janela nova nunca sobrescreve a antiga — os resultados já calculados apontam
para `predep_common_window.nc` (1996–2024).

Mesmo esquema do notebook, para `src/modal/run_bacias.py` ler sem alteração:

    pr_xavier, pr_chirps   (time, pixel)     mm/mês, float32 — totais, não anomalias
    mov                    (time, mov_name)  os 29 índices
    mov_available          (time, mov_name)  bool
    complete_all           (time)            bool — todos os MoVs presentes

Uso:
  python -m src.data_processing.janela_comum --inicio 1982-01 --fim 2024-12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XAVIER_PADRAO = "data/processed/xavier_monthly/pr_monthly_1961_2024_BRAZIL_XAVIER.nc"
CHIRPS_PADRAO = "data/processed/chirps_monthly/pr_monthly_1981_2026_BRAZIL_CHIRPS.nc"
MOVS_PADRAO = "data/raw/mov_indices"
SAIDA_DIR = "data/processed/common_window"


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def carregar_movs(mov_dir) -> pd.DataFrame:
    """Os 27 CSVs + MGI e ATL3, que só existem dentro do .txt do ATL3.

    TNA e TSA também estão no .txt, mas com outra climatologia de referência
    (r = 0,92 e 0,94 contra os CSVs, viés de +0,30 e +0,27). Ficam os dos CSVs,
    como no notebook, para a janela nova ser comparável à antiga.
    """
    mov_dir = _abs(mov_dir)
    out = {}
    for f in sorted(mov_dir.glob("*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        idx = pd.PeriodIndex(pd.to_datetime(d[d.columns[0]]), freq="M")
        out[f.name.split("_")[0]] = pd.Series(
            pd.to_numeric(d[d.columns[1]], errors="coerce").values,
            index=idx).sort_index()
    txt = mov_dir / "ATL3_indice_data.txt"
    if txt.exists():
        a = pd.read_csv(txt, sep=r"\s+", skiprows=11,
                        names=["Year", "Month", "TNA", "TSA", "MGI", "ATL3"])
        a = a.set_index(pd.PeriodIndex(year=a.Year, month=a.Month, freq="M"))
        for col in ("MGI", "ATL3"):
            out[col] = a[col].astype(float)
    return pd.DataFrame(dict(sorted(out.items()))).sort_index()


def construir(inicio: str, fim: str, *, xavier=XAVIER_PADRAO, chirps=CHIRPS_PADRAO,
              movs=MOVS_PADRAO, saida=None) -> Path:
    """Monta e grava a janela. Devolve o caminho do NetCDF."""
    import xarray as xr

    janela = pd.period_range(inicio, fim, freq="M")
    M = carregar_movs(movs)
    xav_ds = xr.open_dataset(_abs(xavier))
    chi_ds = xr.open_dataset(_abs(chirps))
    xav_t = pd.PeriodIndex(pd.DatetimeIndex(xav_ds.time.values), freq="M")
    chi_t = pd.PeriodIndex(pd.DatetimeIndex(chi_ds.time.values), freq="M")

    # A janela precisa estar inteira nas duas fontes: completar com NaN criaria
    # meses sem chuva que o alinhamento trataria como amostra válida ausente.
    for nome, t in (("Xavier", xav_t), ("CHIRPS", chi_t)):
        falta = janela.difference(t)
        if len(falta):
            raise ValueError(f"{nome} não cobre a janela: faltam {len(falta)} meses "
                             f"({falta.min()} .. {falta.max()}); cobre {t.min()} .. {t.max()}")

    print(f"janela {janela.min()} .. {janela.max()} · {len(janela)} meses · "
          f"{len(janela) // 4} por estação")
    incompletos = {}
    for c in M.columns:
        v = M[c].reindex(janela).notna()
        if not v.all():
            falt = janela[~v.values]
            so_dez = set(falt.month) == {12}
            incompletos[c] = f"{len(falt)} meses" + (" (só dezembros)" if so_dez else
                                                     f" ({falt.min()} .. {falt.max()})")
    print(f"MoVs: {M.shape[1]} · completos {M.shape[1] - len(incompletos)}")
    for c, txt in incompletos.items():
        print(f"   {c:<8} faltam {txt}")

    xav = xav_ds["pr"].isel(time=xav_t.isin(janela)).values.astype("float32")
    chi = chi_ds["pr"].isel(time=chi_t.isin(janela)).values.astype("float32")
    assert xav.shape[0] == chi.shape[0] == len(janela)

    # A grade do CHIRPS foi ajustada aos centros do Xavier no builder mensal, então
    # é subconjunto exato dele. Conferido em vez de assumido.
    xlat, xlon = xav_ds.latitude.values, xav_ds.longitude.values
    clat, clon = chi_ds.latitude.values, chi_ds.longitude.values
    lat_sel, lon_sel = np.isin(xlat, clat), np.isin(xlon, clon)
    if not (np.array_equal(xlat[lat_sel], clat) and np.array_equal(xlon[lon_sel], clon)):
        raise ValueError("grade do CHIRPS não é subconjunto ordenado da grade do Xavier")
    xav = xav[:, lat_sel][:, :, lon_sel]

    both = np.isfinite(xav).all(axis=0) & np.isfinite(chi).all(axis=0)
    ii, jj = np.where(both)
    print(f"pixels válidos nas duas fontes em todos os meses: {both.sum():,}")

    ds = xr.Dataset(
        {"pr_xavier": (("time", "pixel"), xav[:, both]),
         "pr_chirps": (("time", "pixel"), chi[:, both]),
         "mov": (("time", "mov_name"), M.reindex(janela).values.astype("float32")),
         "mov_available": (("time", "mov_name"), M.reindex(janela).notna().values),
         "complete_all": ("time", M.reindex(janela).notna().all(axis=1).values)},
        coords={"time": janela.to_timestamp(),
                "pixel": np.arange(int(both.sum())),
                "latitude": ("pixel", xlat[lat_sel][ii].astype("float32")),
                "longitude": ("pixel", xlon[lon_sel][jj].astype("float32")),
                "mov_name": list(M.columns)})
    ds["pr_xavier"].attrs.update(units="mm/month", source=_abs(xavier).name)
    ds["pr_chirps"].attrs.update(units="mm/month", source=_abs(chirps).name)
    ds.attrs.update({
        "title": "Janela comum Xavier ∩ CHIRPS ∩ MoVs para PREDEP/r²",
        "time_window": f"{janela.min()} .. {janela.max()}",
        "strict_months": int(ds["complete_all"].values.sum()),
        "built_by": "src/data_processing/janela_comum.py",
        "precipitacao": "totais mensais brutos (mm/mês); anomalias, se desejadas, "
                        "são calculadas na pipeline",
        "movs_incompletos": "; ".join(f"{c}: {t}" for c, t in incompletos.items()) or "nenhum",
    })

    saida = _abs(saida) if saida else ROOT / SAIDA_DIR / (
        f"predep_janela_{janela.min().year}_{janela.max().year}.nc")
    saida.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ("pr_xavier", "pr_chirps", "mov")}
    ds.to_netcdf(saida, encoding=enc)
    print(f"\n✅ {saida}  ({saida.stat().st_size / 1e6:.0f} MB)")
    print(f"   {ds.sizes['time']} meses · {ds.sizes['pixel']:,} pixels · "
          f"{ds.sizes['mov_name']} MoVs · meses com todos os MoVs: {ds.attrs['strict_months']}")
    return saida


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inicio", default="1982-01")
    ap.add_argument("--fim", default="2024-12")
    ap.add_argument("--xavier", default=XAVIER_PADRAO)
    ap.add_argument("--chirps", default=CHIRPS_PADRAO)
    ap.add_argument("--movs", default=MOVS_PADRAO)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    construir(a.inicio, a.fim, xavier=a.xavier, chirps=a.chirps, movs=a.movs, saida=a.saida)


if __name__ == "__main__":
    main()
