#!/bin/bash

# upload_to_pypi.sh - Upload ins_pricing package to PyPI
# Usage: ./upload_to_pypi.sh
#
# Prerequisites:
#   - Set TWINE_PASSWORD environment variable with your PyPI token
#   - Run `python -m build` to build the distribution files

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Extract version from pyproject.toml
get_version() {
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found in current directory"
        exit 1
    fi

    # Extract version using grep and sed
    VERSION=$(grep '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/')

    if [ -z "$VERSION" ]; then
        print_error "Could not extract version from pyproject.toml"
        exit 1
    fi

    echo "$VERSION"
}

# Get package version
VERSION=$(get_version)
PACKAGE_NAME="ins_pricing"
WHEEL_FILE="dist/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
TARBALL_FILE="dist/${PACKAGE_NAME}-${VERSION}.tar.gz"

echo "========================================"
print_info "Uploading ${PACKAGE_NAME} ${VERSION} to PyPI"
echo "========================================"
echo

# Check if TWINE_PASSWORD is set
if [ -z "$TWINE_PASSWORD" ]; then
    print_error "TWINE_PASSWORD environment variable is not set"
    echo "Please set it with your PyPI token:"
    echo "  export TWINE_PASSWORD='your_token_here'"
    echo
    echo "Or set it inline:"
    echo "  TWINE_PASSWORD='your_token' ./upload_to_pypi.sh"
    exit 1
fi

# Check if wheel file exists
if [ ! -f "$WHEEL_FILE" ]; then
    print_error "Wheel file not found: $WHEEL_FILE"
    echo "Please run: python -m build"
    exit 1
fi

# Check if tarball file exists
if [ ! -f "$TARBALL_FILE" ]; then
    print_error "Source tarball not found: $TARBALL_FILE"
    echo "Please run: python -m build"
    exit 1
fi

# Display files to be uploaded
echo "Files to upload:"
echo "  - $WHEEL_FILE"
echo "  - $TARBALL_FILE"
echo

# Confirm upload (optional, comment out for automation)
# read -p "Proceed with upload? [y/N] " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     print_warning "Upload cancelled by user"
#     exit 0
# fi

print_info "Uploading to PyPI..."
echo

# Upload to PyPI
if python -m twine upload -r pypi -u __token__ -p "$TWINE_PASSWORD" "$WHEEL_FILE" "$TARBALL_FILE"; then
    echo
    echo "========================================"
    print_success "SUCCESS! Package uploaded to PyPI"
    echo "========================================"
    echo
    echo "You can now install it with:"
    print_info "  pip install --upgrade ${PACKAGE_NAME}"
    echo
    echo "View on PyPI:"
    print_info "  https://pypi.org/project/${PACKAGE_NAME}/${VERSION}/"
    exit 0
else
    EXIT_CODE=$?
    echo
    echo "========================================"
    print_error "FAILED! Upload failed with exit code $EXIT_CODE"
    echo "========================================"
    echo
    echo "Common issues:"
    echo "  - Version $VERSION already exists on PyPI"
    echo "  - Invalid TWINE_PASSWORD token"
    echo "  - Network connection issues"
    exit $EXIT_CODE
fi
