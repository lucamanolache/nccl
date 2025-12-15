#!/bin/bash
# Quick fix script for build issues

set -e

echo "=========================================="
echo "Fixing NCCL Build Configuration"
echo "=========================================="
echo ""

# Get NCCL root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

echo "NCCL root directory: $NCCL_ROOT"

# Check if NCCL_HOME is set
if [ -z "$NCCL_HOME" ]; then
    echo "NCCL_HOME not set. Setting to NCCL root..."
    export NCCL_HOME="$NCCL_ROOT"
    echo "NCCL_HOME=$NCCL_HOME"
else
    echo "NCCL_HOME already set to: $NCCL_HOME"
fi

# Check for nccl.h in various locations
echo ""
echo "Checking for nccl.h..."

if [ -f "$NCCL_HOME/include/nccl.h" ]; then
    echo "✓ Found: $NCCL_HOME/include/nccl.h"
elif [ -f "$NCCL_ROOT/src/nccl.h" ]; then
    echo "✓ Found: $NCCL_ROOT/src/nccl.h"
    echo "  Creating include directory and symlink..."
    mkdir -p "$NCCL_HOME/include" 2>/dev/null || true
    if [ ! -f "$NCCL_HOME/include/nccl.h" ]; then
        ln -sf "$NCCL_ROOT/src/nccl.h" "$NCCL_HOME/include/nccl.h" 2>/dev/null || \
        cp "$NCCL_ROOT/src/nccl.h" "$NCCL_HOME/include/nccl.h" 2>/dev/null || true
    fi
    echo "✓ Created: $NCCL_HOME/include/nccl.h"
else
    echo "✗ nccl.h not found!"
    echo ""
    echo "Building NCCL first..."
    cd "$NCCL_ROOT"
    if [ -f "Makefile" ]; then
        echo "Running 'make' in $NCCL_ROOT..."
        make
        # After build, nccl.h should be in src/
        if [ -f "src/nccl.h" ]; then
            mkdir -p "$NCCL_HOME/include" 2>/dev/null || true
            ln -sf "$NCCL_ROOT/src/nccl.h" "$NCCL_HOME/include/nccl.h" 2>/dev/null || \
            cp "$NCCL_ROOT/src/nccl.h" "$NCCL_HOME/include/nccl.h"
            echo "✓ Created: $NCCL_HOME/include/nccl.h"
        fi
    else
        echo "ERROR: No Makefile found in $NCCL_ROOT"
        echo "Please build NCCL first or set NCCL_HOME to installed NCCL location"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

echo ""
echo "=========================================="
echo "Build Configuration Fixed"
echo "=========================================="
echo ""
echo "NCCL_HOME=$NCCL_HOME"
echo ""
echo "Now you can build the example:"
echo "  make clean"
echo "  make"
echo ""
echo "Or run the full test:"
echo "  ./run_and_debug.sh"
echo ""
