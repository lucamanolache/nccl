# Quick Start: Testing AlltoAllV

## Quick Test Commands

### Build
```bash
cd examples/05_symmetric_memory/02_alltoallv
make
```

### Run Test (Public API)
```bash
# Using pthreads (default, uses all available GPUs)
./alltoallv_sm

# Or use make test
make test

# Specify number of GPUs
NTHREADS=4 ./alltoallv_sm
```

### Run with MPI
```bash
make MPI=1
mpirun -np 4 ./alltoallv_sm
```

### Run with Debug Output
```bash
NCCL_DEBUG=INFO ./alltoallv_sm
```

## What Gets Tested

✅ **Public API**: `ncclAlltoAllV()` function call  
✅ **Symmetric Memory**: Window registration and usage  
✅ **CE Backend**: CUDA Engine collectives (if available)  
✅ **Data Verification**: Correctness of variable-sized transfers  
✅ **Multi-rank**: Works with 2+ GPUs/ranks  

## Expected Success Output

```
Starting AlltoAllV example with N ranks
  Rank X communicator initialized using device Y
Setting up AlltoAllV with variable-sized transfers
  Send counts per rank: 1024 512 256 ...
Symmetric Memory allocation
Starting AlltoAllV with variable-sized transfers
AlltoAllV completed successfully
Rank X verification:
  Received from rank 0: X elements at offset Y - OK
  ...
Rank X: Results verified correctly
All resources cleaned up successfully
```

## Troubleshooting

**Problem**: "Symmetric memory not supported"  
**Solution**: Ensure CUDA 12.5+ and compatible GPUs

**Problem**: "CE not implemented"  
**Solution**: Check `NCCL_CTA_POLICY=0` and CUDA driver version

**Problem**: Verification failures  
**Solution**: Check that all ranks have matching send/recv counts

For more details, see [TESTING.md](TESTING.md)
