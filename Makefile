PYTHON = python
SETUP = $(PYTHON) setup.py

.PHONY: build clean dist install

all: build

install: dist
	$(SETUP) install

build:
	$(SETUP) build

dist: build
	$(SETUP) sdist

clean:
	$(SETUP) clean --all
