PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: deck

# One figure per 16:9 slide, order from paper/figures_manifest.csv -> slides/deck_<date>.pptx
deck:
	$(PYTHON) scripts/make_deck.py
