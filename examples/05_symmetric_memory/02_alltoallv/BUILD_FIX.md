# Build Fix: ncclWindow_t Not Found

## Problem
The compiler can't find `ncclWindow_t` and related definitions because:
1. NCCL needs to be built first (generates `nccl.h` from `nccl.h.in`)
2. `NCCL_HOME` environment variable needs to point to NCCL build directory

## Solution

### Step 1: Build NCCL First

```bash
# From the NCCL root directory
cd /path/to/nccl  # or cd ../../.. from examples/05_symmetric_memory/02_alltoallv

# Build NCCL (this generates nccl.h)
make

# Or if you have a build directory
cd build
make
```

### Step 2: Set NCCL_HOME

```bash
# If NCCL is in the source tree (typical development setup)
export NCCL_HOME=$(pwd)  # from NCCL root directory

# Or if NCCL is installed elsewhere
export NCCL_HOME=/path/to/nccl/build

# Or if using installed NCCL
export NCCL_HOME=/usr/local  # or wherever NCCL is installed
```

### Step 3: Verify nccl.h Exists

```bash
# Check that nccl.h exists
ls -la $NCCL_HOME/include/nccl.h

# Or if in source tree
ls -la src/nccl.h  # This should exist after building
```

### Step 4: Build the Example

```bash
cd examples/05_symmetric_memory/02_alltoallv

# Make sure NCCL_HOME is set
echo $NCCL_HOME

# Build
make clean
make
```

## Alternative: Point to Source Directory

If NCCL isn't installed but you're building from source:

```bash
# From NCCL root directory
export NCCL_HOME=$(pwd)

# The Makefile will look for:
# - $NCCL_HOME/include/nccl.h (for installed)
# - $NCCL_HOME/src/nccl.h (might work if include path is adjusted)

# Actually, you may need to adjust the include path
# Check what the Makefile expects:
cd examples/05_symmetric_memory/02_alltoallv
make INCLUDES="-I../../../src -I$(CUDA_HOME)/include" 
```

## Quick Fix Script

Create this script and run it:

```bash
#!/bin/bash
# fix_build.sh

# Get NCCL root directory (assuming we're in examples/...)
NCCL_ROOT=$(cd ../../.. && pwd)
export NCCL_HOME=$NCCL_ROOT

echo "NCCL_HOME set to: $NCCL_HOME"
echo "Checking for nccl.h..."

if [ -f "$NCCL_HOME/src/nccl.h" ]; then
    echo "Found: $NCCL_HOME/src/nccl.h"
    # Create symlink or adjust include
    mkdir -p $NCCL_HOME/include 2>/dev/null
    ln -sf $NCCL_HOME/src/nccl.h $NCCL_HOME/include/nccl.h 2>/dev/null || true
elif [ -f "$NCCL_HOME/include/nccl.h" ]; then
    echo "Found: $NCCL_HOME/include/nccl.h"
else
    echo "ERROR: nccl.h not found!"
    echo "Building NCCL first..."
    cd $NCCL_ROOT
    make
    cd -
fi

echo "Now building example..."
make clean
make
```

## Verify Build Setup

```bash
# Check include paths
make -n 2>&1 | grep "^-I"

# Should show something like:
# -I/path/to/nccl/include -I/usr/local/cuda/include
```

## If Still Failing

Try building with explicit paths:

```bash
cd examples/05_symmetric_memory/02_alltoallv

# Find where nccl.h actually is
find ../../.. -name "nccl.h" -type f

# Then set NCCL_HOME to that directory's parent
# For example, if found at ../../../src/nccl.h:
export NCCL_HOME=$(cd ../../.. && pwd)
make INCLUDES="-I$NCCL_HOME/src -I$CUDA_HOME/include"
```
