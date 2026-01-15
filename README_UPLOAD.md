# PyPI Upload Scripts Usage Guide

This project provides two upload scripts for different operating systems:

## Windows Users

Use `upload_to_pypi.bat` (auto-reads version from pyproject.toml):

```cmd
# 1. Set PyPI Token
set TWINE_PASSWORD=your_pypi_token_here

# 2. Build distribution packages
python -m build

# 3. Upload to PyPI (auto-detects version)
upload_to_pypi.bat
```

## Linux / macOS Users

Use `upload_to_pypi.sh`:

```bash
# 1. Set executable permission (first time only)
chmod +x upload_to_pypi.sh

# 2. Set PyPI Token
export TWINE_PASSWORD='your_pypi_token_here'

# 3. Build distribution packages
python -m build

# 4. Upload to PyPI
./upload_to_pypi.sh
```

### One-liner Execution

```bash
TWINE_PASSWORD='your_token' ./upload_to_pypi.sh
```

## Features

### Common Features
- ✅ Auto-read version from `pyproject.toml`
- ✅ Validate distribution files exist
- ✅ Check environment variables
- ✅ Detailed success/failure feedback

### Shell Script Extra Features
- 🎨 Colored output (error/success/info)
- 🔒 Auto-exit on error (`set -e`)
- 📦 Display file list to be uploaded
- 🔗 Provide PyPI package page link
- 💡 Common error hints

## Prerequisites

### Install Build Tools

```bash
pip install --upgrade build twine
```

### Get PyPI Token

1. Login to [PyPI](https://pypi.org/)
2. Go to Account Settings → API tokens
3. Create new token (select "Entire account" or specific project)
4. Save token (format: `pypi-...`)

### Environment Variable Setup

#### Linux / macOS (Persistent)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export TWINE_PASSWORD='pypi-your-token-here'
```

Then execute:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

#### Windows (Persistent)

Via system settings or command line:

```cmd
setx TWINE_PASSWORD "pypi-your-token-here"
```

## Complete Workflow

```bash
# 1. Update version number
# Edit the version field in pyproject.toml

# 2. Build distribution packages
python -m build

# 3. Check built packages
python -m twine check dist/*

# 4. (Optional) Upload to TestPyPI for testing
python -m twine upload --repository testpypi dist/*

# 5. Upload to PyPI
./upload_to_pypi.sh  # Linux/macOS
# or
upload_to_pypi.bat   # Windows
```

## Troubleshooting

### Version Already Exists Error

```
HTTPError: 400 Bad Request - File already exists
```

**Solution**: Update version in `pyproject.toml`, then rebuild.

### Invalid Token Error

```
HTTPError: 403 Forbidden - Invalid or non-existent authentication
```

**Solution**:
1. Check if `TWINE_PASSWORD` is set correctly
2. Ensure token includes `pypi-` prefix
3. Verify token hasn't expired

### File Not Found Error

```
ERROR: Wheel file not found
```

**Solution**: Run `python -m build` to build distribution packages.

## Security Recommendations

- ⚠️ **DO NOT** commit Token to version control
- ✅ Use environment variables or secret management tools to store Token
- ✅ Rotate PyPI Token regularly
- ✅ Use different scoped tokens for different projects

## Script Comparison

| Feature | upload_to_pypi.bat | upload_to_pypi.sh |
|---------|-------------------|-------------------|
| Version extraction | ✅ Auto from pyproject.toml | ✅ Auto from pyproject.toml |
| Colored output | ❌ | ✅ |
| Error handling | Basic | Enhanced (set -e) |
| File list display | ✅ | ✅ |
| PyPI link | ✅ | ✅ |
| Error hints | ✅ | ✅ |

**Both scripts now have feature parity, with the main difference being colored output support.**

## Changelog

- **v2.1** (2026-01-15) - Windows batch script now supports auto version extraction, both platforms feature parity
- **v2.0** - Shell script supports auto version extraction and colored output
- **v1.0** - Initial Windows batch script (hardcoded version)
