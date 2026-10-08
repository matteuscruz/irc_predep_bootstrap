# Dados

O que está versionado e o que precisa ser obtido fora do git. Os checksums SHA-256 de
todos os arquivos usados estão em [`checksums_sha256.txt`](checksums_sha256.txt);
confira uma cópia com `sha256sum -c data/checksums_sha256.txt` (a partir da raiz).

## No git

| caminho | tamanho | conteúdo |
|---|---|---|
| `raw/mov_indices/` | 848 kB | os 29 índices de MoV como recebidos (27 CSV da base UNIFEI, `ATL3_indice_data.txt` do arquivo de índices atlânticos NOAA/PSL que acompanha a base, e o `movs_data.zip` original) |
| `clusters/` | — | shapefiles das 12 bacias hidrográficas (máscara ponto-em-polígono) |
| `reference/` | — | contornos da América do Sul para os mapas (`south_america_110m.geojson`) |
| `processed/common_window/predep_janela_1982_2024.nc` | 213 MB (LFS) | **entrada única do pipeline**: precipitação mensal Xavier e CHIRPS na grade comum, 1982-01 a 2024-12 (516 meses × 70.191 células), e os 29 MoVs alinhados |

Com a janela comum, todo o PREDEP, r² e SOM do artigo podem ser refeitos sem os dados
brutos (README da raiz, seções 3 e 9). Ela foi gerada por
`tools/build_monthly_from_raw.py` (mensais) e `src/data_processing/janela_comum.py`.
Variáveis: `pr_xavier`, `pr_chirps` (time, pixel; mm/mês, totais brutos — as anomalias
são calculadas no pipeline), `mov` e `mov_available` (time, mov_name), `complete_all`.
Comprimida sem perda (zlib 5, shuffle).

Particularidades dos índices, tratadas no pipeline: EPNP não tem nenhum dezembro;
IOSD e SIOD são idênticos; TNA e TSA existem em duas versões (usada a distribuída
individualmente); IASAS, MGI e ITSMRG2 têm definição ainda não confirmada.

## Fora do git (dados brutos, ~22 GB)

Dados públicos; não versionados por tamanho (vários arquivos acima do limite de 2 GB
por arquivo do Git LFS). As datas são as dos arquivos locais, não necessariamente as
do download.

| caminho local | tamanho | produto | fonte |
|---|---|---|---|
| `raw/xavier_daily/pr_*_BR-DWGD_UFES_UTEXAS_v_3.2.2-00{1,2,3,4}.nc` | 11,9 GB | BR-DWGD v3.2.2, precipitação diária 0,1°, 1961–2024 (o `-004` cobre 2020-01-01 a 2024-12-31 apesar do nome) | Xavier et al. (2022), doi:10.1002/joc.7731; atualizações: https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data |
| `raw/chirps_daily/chirps_v2_brazil.zarr` | 4,6 GB | CHIRPS 2.0 diário 0,05°, recorte Brasil (zarr v3) — usado no pipeline | Funk et al. (2015); https://data.chc.ucsb.edu/products/CHIRPS-2.0/ ; documentação: https://pubs.usgs.gov/ds/832/ |
| `raw/chirps_daily/chirps_v2.nc` | 486 MB | CHIRPS 2.0, só a bacia do Paraná (versão antiga; não usada nos resultados atuais) | idem |
| `raw/chirps.zip` | 5,0 GB | cópia compactada do CHIRPS baixado | idem |
| `raw/xavier_daily/pr_monthly_1961_2019_BRAZIL_combined.nc` | 870 MB | mensal combinado antigo (até 2019); não usado — o mensal atual é gerado do diário | — |

Datas dos arquivos locais: BR-DWGD 2026-02-03; CHIRPS 2026-09-14.

**Atenção no BR-DWGD `-004`:** a partir de 2024-03-21 o arquivo deixa de aplicar a
máscara continental e a interpolação vaza para o oceano; `build_monthly_from_raw.py`
reaplica a máscara dos arquivos anteriores. Os metadados do arquivo dizem "Created
2024-09-02", embora ele contenha dados até dezembro de 2024 — a confirmar com os
autores.

## Arquivamento recomendado

Para o artigo, depositar no Zenodo (gratuito até 50 GB, com DOI) os arquivos brutos
usados (`xavier_daily/*-00{1,2,3,4}.nc`, `chirps_v2_brazil.zarr` compactado) e a
janela comum, junto com `checksums_sha256.txt`, e citar o DOI na seção de dados.

## Outros

`fatias_modal/` (fatias do Modal, fora do git) não é necessário para reproduzir o
artigo a partir da janela comum.

`input/` é do pipeline antigo (anomalias **sazonais** 1982–2020 e
`merged_indices.csv`), usado pelos experimentos em `experiments/`; não entra nos
resultados atuais. A versão anterior deste README, que descrevia esse formato, está
no histórico do git (`git show a886012:data/README.md`).
