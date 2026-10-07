#!/bin/sh
# SOMs nacionais por defasagem que faltam (2..6), dois de cada vez.
cd "$(dirname "$0")/.."
seq 2 6 | xargs -P 2 -I{} sh -c 'PYTHONUNBUFFERED=1 .venv/bin/python -u tools/brasil_local.py --n-boot 10000 --pular-juntar --lag-fixo {} > logs/som_lag{}.log 2>&1; echo "lag {} terminou (exit $?)" >> logs/som_lags.status'
.venv/bin/python tools/resumo_som_lags.py --n-boot 10000 > logs/som_lags_resumo.log 2>&1
echo "TUDO PRONTO" >> logs/som_lags.status
