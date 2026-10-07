cd /media/matteuscruz/Projects/Entrep/irc_predep_bootstrap
.venv/bin/python tools/som_consenso_local.py --n-boot 10000 --entrada brasil_lags.nc --com-ano --metrica alpha --n-proc 5
.venv/bin/python tools/som_consenso_local.py --n-boot 10000 --entrada brasil_lags.nc --com-ano --metrica r2 --n-proc 5
