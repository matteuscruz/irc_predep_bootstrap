# Figuras do artigo a partir dos .nc prontos (nenhum cálculo de PREDEP ou SOM).
#
#   make            # todas as figuras
#   make maximo     # só os mapas de máximo (PREDEP e r²)
#   make copiar     # copia as figuras para a pasta do artigo (PAPER_FIGS)
#
# Variáveis: PYTHON, DADOS, SAIDA, PAPER_FIGS — ex.: make PYTHON=python3 DADOS=/outro/lugar

RAIZ       := $(abspath ..)
PYTHON     ?= $(RAIZ)/.venv/bin/python
DADOS      ?= $(CURDIR)/dados
SAIDA      ?= $(CURDIR)/figuras
PAPER_FIGS ?= $(RAIZ)/predep-climate-variability/figs

export DADOS SAIDA
RODA := cd $(CURDIR) && $(PYTHON)

.PHONY: all estimador maximo predep_max r2_max som sensibilidade fluxograma copiar limpar

all: estimador maximo som sensibilidade fluxograma

estimador:
	$(RODA) fig_nboot.py

maximo: predep_max r2_max

predep_max:
	$(RODA) fig_maximo.py --metrica alpha

r2_max:
	$(RODA) fig_maximo.py --metrica r2

som:
	$(RODA) fig_som.py --metrica alpha
	$(RODA) fig_som.py --metrica r2
	$(RODA) fig_som.py --metrica alpha --intersecao
	$(RODA) fig_som.py --metrica r2 --intersecao

sensibilidade:
	$(RODA) fig_sensibilidade_k.py

fluxograma:
	mkdir -p $(SAIDA)/fluxograma
	cd fluxograma && pdflatex -interaction=nonstopmode -output-directory=$(SAIDA)/fluxograma \
		fluxograma_standalone.tex > /dev/null
	rm -f $(SAIDA)/fluxograma/*.aux $(SAIDA)/fluxograma/*.log $(SAIDA)/fluxograma/*.out
	@echo "  ✅ $(SAIDA)/fluxograma/fluxograma_standalone.pdf"

copiar:
	for d in estimador predep_max r2_max som; do \
		mkdir -p $(PAPER_FIGS)/$$d && cp $(SAIDA)/$$d/*.png $(PAPER_FIGS)/$$d/; done
	mkdir -p $(PAPER_FIGS)/fluxograma && cp fluxograma/fluxograma.tex $(PAPER_FIGS)/fluxograma/
	@echo "  figuras copiadas para $(PAPER_FIGS)"

limpar:
	rm -rf $(SAIDA)
