PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: deck

# One figure per 16:9 slide, order from paper/figures_manifest.csv -> slides/deck_<date>.pptx
deck:
	$(PYTHON) scripts/make_deck.py

# Research journal (Quarto, installed in ~/opt; see README.md "Журнал")
QUARTO ?= $(shell command -v quarto 2>/dev/null || ls -d $(HOME)/opt/quarto-*/bin/quarto 2>/dev/null | tail -1)
DATE := $(shell date +%Y%m%d)

.PHONY: journal journal-auto journal-preview journal-pptx journal-docx journal-pdf

journal-auto:
	$(PYTHON) scripts/journal_build_auto.py

journal: journal-auto
	$(QUARTO) render journal
	@echo "-> journal/_site/index.html"

journal-preview: journal-auto
	$(QUARTO) preview journal

journal-pptx: journal-auto
	mkdir -p slides
	$(QUARTO) render journal/slides.qmd --to pptx --output deck_$(DATE).pptx
	mv deck_$(DATE).pptx slides/deck_$(DATE).pptx
	@echo "-> slides/deck_$(DATE).pptx"

journal-docx: journal-auto
	$(QUARTO) render journal --to docx
	@echo "-> journal/_site/**/*.docx"

journal-pdf: journal-auto
	$(QUARTO) render journal --to typst
	@echo "-> journal/_site/**/*.pdf (Typst)"
