# Figuras do artigo

Um script por figura do artigo. Todos só **leem resultados prontos** (`.nc`) e
desenham — nenhum recalcula PREDEP, r² ou SOM. Para gerar os `.nc`, veja a seção
"9. Produtos finais do paper" do README da raiz.

```bash
cd plots_artigo
make            # todas as figuras → plots_artigo/figuras/
make copiar     # copia para predep-climate-variability/figs/
```

## Figuras

| alvo `make` | script | figura no artigo | entrada (em `DADOS`) | saída (em `SAIDA`) |
|---|---|---|---|---|
| `estimador` | `fig_nboot.py` | efeito de n_b no estimador | — (valores medidos fixos em `tools/diagnostico_ruido_predep.py`) | `estimador/fig_nboot.png` |
| `predep_max` | `fig_maximo.py --metrica alpha` | PREDEP máximo por estação e lag | `max_lags/predep_max_lags.nc` | `predep_max/predep_max_{djf,mam,jja,son}.png` |
| `r2_max` | `fig_maximo.py --metrica r2` | r² máximo por estação e lag | `max_lags/r2_max_lags.nc` | `r2_max/r2_max_{djf,mam,jja,son}.png` |
| `som` | `fig_som.py --metrica alpha\|r2 [--intersecao]` | regimes do SOM de consenso | `som/som_predep.nc`, `som/som_r2.nc` | `som/som_{predep,r2}[_intersecao].png` |
| `sensibilidade` | `fig_sensibilidade_k.py` | sensibilidade a K | `som/sensibilidade_k_{alpha,r2}.nc` | `som/sensibilidade_k.png` |
| `fluxograma` | `fluxograma/fluxograma.tex` (TikZ) | fluxo metodológico | — | `fluxograma/fluxograma_standalone.pdf` |

`--intersecao`: cada regime lista só os MoVs que se destacam na assinatura de Xavier
**e** na de CHIRPS. Sem ela, a lista vem da assinatura média das duas fontes.

No artigo, o fluxograma entra como código (`\input{figs/fluxograma/fluxograma.tex}`,
com `tikz` e as bibliotecas `positioning,arrows.meta,fit,backgrounds`); o PDF isolado
serve só para conferência.

## Dados de entrada

Uma cópia dos `.nc` usados pelas figuras está em `plots_artigo/dados/` (37 MB,
versionada via Git LFS), então a pasta é autossuficiente para gerar as figuras:

```
dados/
├── max_lags/
│   ├── predep_max_lags.nc        17 MB   PREDEP máximo por estação × lag × célula
│   └── r2_max_lags.nc            16 MB   o mesmo para o r²
├── cubo/                         ~1 GB   PREDEP e r² completos (só local, fora do git)
└── som/
    ├── som_predep.nc             2,7 MB  SOM de consenso sobre o PREDEP (α)
    ├── som_r2.nc                 3,0 MB   SOM de consenso sobre o r²
    ├── sensibilidade_k_alpha.nc  16 kB   métricas por K, PREDEP
    └── sensibilidade_k_r2.nc     16 kB   métricas por K, r²
```

### Cubo completo (só local)

`dados/cubo/` tem os valores espaciais completos — o PREDEP e o r² de **cada MoV**, em
cada fonte, estação, lag e célula:

| arquivo | tamanho | variável e dimensões |
|---|---|---|
| `cubo/predep_brasil.nc` | 478 MB | `alpha` (source=xavier,chirps × season × lag=0–6,8,10,12 × mov_name=29 × pixel=70.189) |
| `cubo/r2_brasil.nc` | 529 MB | `r2`, mesmas dimensões |

É a base de onde saem todos os outros arquivos (máximos, SOM). Não vai para o git:
cerca de 1 GB excede o limite de arquivo e a cota gratuita de LFS do GitHub; está em
`plots_artigo/.gitignore`. Original: `artifacts/<config>/raw/basins/brasil_lags.nc`
(`tools/juntar_lags.py`). Para lê-lo: `xr.open_dataset("dados/cubo/r2_brasil.nc").r2`.

### Arquivos das figuras

| arquivo | variáveis principais | gerado por | original |
|---|---|---|---|
| `{predep,r2}_max_lags.nc` | `valor_max`, `mov_max`, `dif_fontes` (estação, lag, célula) | `tools/fig_predep_max_lags.py` | `artifacts/<config>/plots/brasil/predep_max_lags/` |
| `som_{predep,r2}.nc` | `regime`, `confianca`, `assinatura` (por fonte), `concord_assinatura` (ρ), `estab_consenso`, `sel_lag`, `sel_mov` | `tools/som_consenso_local.py` | `artifacts/<config>/som/brasil_consenso_todos_lags[_r2].nc` |
| `sensibilidade_k_{alpha,r2}.nc` | `estab_consenso`, `rho_fontes`, `var_explicada`, `silhouette` (estação, K) | `tools/som_sensibilidade_k.py` | gerado direto em `paper/dados/som/` |

Todos os `.nc` estão comprimidos sem perda (zlib nível 5, com shuffle).

Configuração de todos: `predep_janela_1982_2024_anom_nb10000` — anomalias mensais,
janeiro de 1982 a dezembro de 2024, 70.189 células, 29 MoVs, lags 0–6, 8, 10 e 12,
n_b = 10.000, Xavier (BR-DWGD v3.2.2) e CHIRPS 2.0. O `som_r2.nc` tem também as
partições das 20 sementes (`particoes`); o `som_predep.nc` não, por ser de uma rodada
anterior a essa mudança. A figura do estimador (`fig_nboot`) não lê `.nc`: os
valores medidos estão fixos em `tools/diagnostico_ruido_predep.py`.

Para usar outra pasta de dados: `make DADOS=/outro/lugar`.

## Variáveis

| variável | padrão | uso |
|---|---|---|
| `PYTHON` | `../.venv/bin/python` | interpretador (precisa de numpy, xarray, matplotlib, geopandas) |
| `DADOS` | `./dados` | pasta dos `.nc` de entrada |
| `SAIDA` | `./figuras` | pasta das figuras geradas |
| `PAPER_FIGS` | `../predep-climate-variability/figs` | destino do `make copiar` |

Os scripts também rodam sem o `make`: `python plots_artigo/fig_maximo.py --metrica r2`,
com `DADOS` e `SAIDA` opcionais como variáveis de ambiente.

O fluxograma precisa de `pdflatex` com TikZ.
