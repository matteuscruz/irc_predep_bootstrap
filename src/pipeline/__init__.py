"""
Orquestração ponta a ponta: PREDEP + r² + SOM, por bacia.

`executar_bacia`/`executar_varias` NÃO são reexportados aqui: importá-los
carregaria numpy e scipy só por tocar no pacote, e o despacho ao Modal precisa
importar `pipeline.config` sem isso. Importe direto de `src.pipeline.bacia`.
"""
