# Makefile for SurvRNC

# Variables
PYTHON=python
PIP=pip
DOCKER=docker
DOCKER_IMAGE=survrnc:latest

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help                Show this help message"
	@echo "  install             Install dependencies"
	@echo "  setup-env           Set up the development environment"
	@echo "  run-tests           Run unit tests"
	@echo "  build               Build the project"
	@echo "  build-docker        Build the Docker image"
	@echo "  run-docker          Run the Docker container"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Set up the development environment
.PHONY: setup-env
setup-env: install
	$(PYTHON) -m venv venv
	. venv/bin/activate

# Run unit tests
.PHONY: run-tests
run-tests:
	$(PYTHON) -m unittest discover -s tests

# Build the project
.PHONY: build
build:
	$(PYTHON) setup.py build

# Build the Docker image
.PHONY: build-docker
build-docker:
	$(DOCKER) build -t $(DOCKER_IMAGE) .

# Run the Docker container
.PHONY: run-docker
run-docker:
	$(DOCKER) run -p 80:80 $(DOCKER_IMAGE)
