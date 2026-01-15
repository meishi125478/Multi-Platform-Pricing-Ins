.PHONY: help build check upload upload-test clean install-dev test

# Default target
help:
	@echo "ins_pricing Package Management"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make build        - Build distribution packages (wheel + sdist)"
	@echo "  make check        - Check built packages with twine"
	@echo "  make upload       - Upload to PyPI (requires TWINE_PASSWORD)"
	@echo "  make upload-test  - Upload to TestPyPI (requires TWINE_PASSWORD)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install-dev  - Install package in editable mode with all extras"
	@echo "  make test         - Run tests"
	@echo ""
	@echo "Environment variables:"
	@echo "  TWINE_PASSWORD    - Your PyPI API token (required for upload)"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. make build"
	@echo "  2. make check"
	@echo "  3. make upload-test  # Optional: test on TestPyPI first"
	@echo "  4. make upload"

# Build distribution packages
build:
	@echo "Building distribution packages..."
	python -m build
	@echo ""
	@echo "Build complete! Files created in dist/"
	@ls -lh dist/ 2>/dev/null || dir dist\ 2>NUL

# Check built packages
check:
	@echo "Checking built packages with twine..."
	python -m twine check dist/*

# Upload to PyPI
upload:
	@if [ -z "$$TWINE_PASSWORD" ]; then \
		echo "ERROR: TWINE_PASSWORD not set"; \
		echo "Set it with: export TWINE_PASSWORD='your_token'"; \
		exit 1; \
	fi
	@echo "Uploading to PyPI..."
	./upload_to_pypi.sh

# Upload to TestPyPI (for testing)
upload-test:
	@if [ -z "$$TWINE_PASSWORD" ]; then \
		echo "ERROR: TWINE_PASSWORD not set"; \
		echo "Set it with: export TWINE_PASSWORD='your_token'"; \
		exit 1; \
	fi
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Install in development mode with all extras
install-dev:
	@echo "Installing ins_pricing in editable mode with all extras..."
	pip install -e ".[bayesopt,plotting,explain,geo,gnn,governance,production,reporting,dev]"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Version management
version:
	@grep '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/'

# Display package info
info:
	@echo "Package Information"
	@echo "==================="
	@echo "Name:    ins_pricing"
	@echo "Version: $$(grep '^version = ' pyproject.toml | sed -E 's/version = \"(.*)\"/\1/')"
	@echo ""
	@echo "Distribution files:"
	@ls -1 dist/ 2>/dev/null || dir /B dist\ 2>NUL || echo "  (none - run 'make build' first)"
