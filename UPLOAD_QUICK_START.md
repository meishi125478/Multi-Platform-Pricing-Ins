# PyPI Upload Quick Start Guide

## 🚀 Upload to PyPI in 5 Minutes

### Windows Users

```cmd
# 1. Install dependencies (first time only)
pip install --upgrade build twine

# 2. Set PyPI Token (first time only)
setx TWINE_PASSWORD "pypi-your-token-here"
# Restart command prompt for environment variable to take effect

# 3. Update version number
# Edit pyproject.toml, change version = "0.2.9" to new version

# 4. Build and upload
python -m build
upload_to_pypi.bat
```

### Linux / macOS Users

```bash
# 1. Install dependencies (first time only)
pip install --upgrade build twine

# 2. Set executable permission (first time only)
chmod +x upload_to_pypi.sh

# 3. Set PyPI Token (first time only)
echo "export TWINE_PASSWORD='pypi-your-token-here'" >> ~/.bashrc
source ~/.bashrc

# 4. Update version number
# Edit pyproject.toml, change version = "0.2.9" to new version

# 5. Build and upload
python -m build
./upload_to_pypi.sh
```

## 🔑 Get PyPI Token

1. Visit https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `ins_pricing_upload`
4. Scope: Select "Project: ins_pricing" or "Entire account"
5. Copy the generated token (format: `pypi-...`)

⚠️ **Important**: Token is shown only once, save it securely!

## ✅ Verify Upload Success

After successful upload, visit:
- **Package page**: https://pypi.org/project/ins_pricing/
- **Specific version**: https://pypi.org/project/ins_pricing/0.2.9/

Test installation:
```bash
pip install --upgrade ins_pricing
python -c "import ins_pricing; print('Installation successful!')"
```

## ❓ Common Issues

### "File already exists" Error
```
HTTPError: 400 Bad Request - File already exists
```
**Solution**: PyPI doesn't allow overwriting published versions. Update version in `pyproject.toml`.

### "Invalid token" Error
```
HTTPError: 403 Forbidden
```
**Solution**:
1. Check if `TWINE_PASSWORD` environment variable is set correctly
2. Ensure token includes `pypi-` prefix
3. Verify token permissions

### "File not found" Error
```
ERROR: Wheel file not found
```
**Solution**: Run `python -m build` to build distribution packages.

## 📋 Complete Workflow

```bash
# 1. Modify code
git add .
git commit -m "feat: add new feature"

# 2. Update version number
# Edit pyproject.toml: version = "0.3.0"

# 3. Build
python -m build

# 4. Check (optional but recommended)
python -m twine check dist/*

# 5. Test upload to TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# 6. Official upload
upload_to_pypi.bat        # Windows
./upload_to_pypi.sh       # Linux/macOS

# 7. Git tag version
git tag v0.3.0
git push origin v0.3.0
```

## 🛠️ Using Makefile (Recommended)

If `make` is installed:

```bash
make build        # Build packages
make check        # Check packages
make upload-test  # Test upload
make upload       # Official upload
make clean        # Clean build artifacts
```

## 📚 Detailed Documentation

See [README_UPLOAD.md](README_UPLOAD.md) for complete documentation.
