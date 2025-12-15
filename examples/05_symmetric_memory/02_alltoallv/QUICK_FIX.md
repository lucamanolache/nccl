# Quick Fix for Build Error

## The Problem
```
error: 'ncclWindow_t' was not declared in this scope
error: 'ncclCommWindowRegister' was not declared
```

This happens because `NCCL_HOME` isn't set or `nccl.h` isn't found.

## Quick Solution (Choose One)

### Option 1: Use the Fix Script (Easiest)
```bash
cd examples/05_symmetric_memory/02_alltoallv
./fix_build.sh
make clean
make
```

### Option 2: Set NCCL_HOME Manually
```bash
# From NCCL root directory
export NCCL_HOME=$(pwd)

# Build NCCL if not already built
make

# Build the example
cd examples/05_symmetric_memory/02_alltoallv
make clean
make
```

### Option 3: Point to Source Directory
```bash
# If NCCL is in source tree (not installed)
export NCCL_HOME=$(cd examples/05_symmetric_memory/02_alltoallv/../../.. && pwd)

# Ensure nccl.h exists (build NCCL first)
cd $NCCL_HOME
make

# Create include directory if needed
mkdir -p $NCCL_HOME/include
ln -sf $NCCL_HOME/src/nccl.h $NCCL_HOME/include/nccl.h

# Build example
cd examples/05_symmetric_memory/02_alltoallv
make clean
make
```

## Verify It Works

```bash
# Check NCCL_HOME is set
echo $NCCL_HOME

# Check nccl.h exists
ls -la $NCCL_HOME/include/nccl.h

# Try building
make clean && make
```

## Still Having Issues?

1. **Make sure NCCL is built:**
   ```bash
   cd ../../..
   make
   ```

2. **Check where nccl.h is:**
   ```bash
   find ../../.. -name "nccl.h" -type f
   ```

3. **Set NCCL_HOME to that location's parent:**
   ```bash
   # If found at ../../../src/nccl.h:
   export NCCL_HOME=$(cd ../../.. && pwd)
   mkdir -p $NCCL_HOME/include
   cp $NCCL_HOME/src/nccl.h $NCCL_HOME/include/nccl.h
   ```

4. **Try building with explicit include:**
   ```bash
   make INCLUDES="-I../../../src -I$CUDA_HOME/include"
   ```

For more details, see [BUILD_FIX.md](BUILD_FIX.md)
