# Testing AlltoAllV API

This document explains how to test both the public API and verify the implementation.

## Prerequisites

1. Build NCCL with CUDA support (CUDA 12.5+ for CE collectives)
2. Have at least 2 GPUs available (or use MPI for multi-node)
3. Ensure symmetric memory is properly configured

## Building the Test

```bash
cd examples/05_symmetric_memory/02_alltoallv
make
```

Or build from the parent directory:
```bash
cd examples/05_symmetric_memory
make -C 02_alltoallv
```

## Running Tests

### Option 1: Using pthreads (Single Node, Multiple GPUs)

This uses one thread per GPU on a single node:

```bash
cd examples/05_symmetric_memory/02_alltoallv
make test
# Or directly:
./alltoallv_sm
```

To specify the number of GPUs:
```bash
NTHREADS=4 ./alltoallv_sm
```

### Option 2: Using MPI (Multi-Node or Single Node)

Build with MPI support:
```bash
cd examples/05_symmetric_memory/02_alltoallv
make MPI=1
```

Run with MPI:
```bash
mpirun -np 4 ./alltoallv_sm
```

### Option 3: Using Make Test Target

The Makefile includes a test target:
```bash
make test
```

This will:
- Build the executable if needed
- Run with all available GPUs (pthread mode) or 2 processes (MPI mode)

## What the Test Does

1. **Initialization**: Sets up NCCL communicator with symmetric memory
2. **Data Setup**: 
   - Creates variable-sized send counts (more data to lower ranks = higher priority)
   - Initializes send buffer with pattern: `(my_rank * 1000 + dest_rank * 100 + element_index)`
3. **AlltoAllV Operation**: Calls `ncclAlltoAllV` public API
4. **Verification**: 
   - Checks that received data matches expected pattern
   - Verifies data from rank `src` is at correct position in receive buffer
   - Validates all elements are correct

## Expected Output

```
Starting AlltoAllV example with N ranks
  Rank X communicator initialized using device Y
Setting up AlltoAllV with variable-sized transfers
  Send counts per rank: 1024 512 256 ...
  Total send elements: X, Total recv elements: Y
Symmetric Memory allocation
  Rank X allocating send: X.XX MB, recv: X.XX MB
Starting AlltoAllV with variable-sized transfers
AlltoAllV completed successfully
Rank X verification:
  Received from rank 0: X elements at offset Y - OK
  ...
Rank X: Results verified correctly
All resources cleaned up successfully
```

## Testing Different Scenarios

### Test with Different Send Patterns

Edit `main.cc` to change the send count calculation:
```cpp
// Current: sendcounts[r] = max_elements_per_rank / (r + 1)
// Try: sendcounts[r] = (r + 1) * 100;  // More data to higher ranks
```

### Test with Zero Counts

Some ranks might send 0 elements:
```cpp
if (r == 2) sendcounts[r] = 0;  // Rank 2 sends nothing
```

### Test with Large Data

Increase `max_elements_per_rank`:
```cpp
size_t max_elements_per_rank = 1024 * 1024;  // 1M elements
```

## Debugging

### Enable NCCL Debug Logging

```bash
NCCL_DEBUG=INFO ./alltoallv_sm
```

### Check if CE is Being Used

```bash
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm
```

### Verify Symmetric Memory Registration

The test will print if symmetric memory windows are registered successfully.

## Troubleshooting

### "Symmetric memory not supported"
- Ensure you're using CUDA 12.5+ with compatible GPUs
- Check that buffers are allocated with `ncclMemAlloc`
- Verify windows are registered with `NCCL_WIN_COLL_SYMMETRIC` flag

### "CE not implemented"
- Check CUDA driver version: `nvidia-smi`
- Ensure `ncclCeImplemented` returns true for AlltoAllV
- Verify `NCCL_CTA_POLICY=0` is set

### Verification Failures
- Check that send/recv counts match across all ranks
- Verify displacements are calculated correctly
- Ensure data patterns match expected values

## Running All Tests

To run all symmetric memory examples:

```bash
cd examples/05_symmetric_memory
make test
```

This will build and test all examples including AlltoAllV.
