# Quick Debug Reference

## 🚀 Fastest Way to Test

```bash
cd examples/05_symmetric_memory/02_alltoallv
./run_and_debug.sh
```

This script will:
1. Check prerequisites
2. Build the test
3. Run it
4. Show debug output
5. Verify success

## 📋 Manual Steps

```bash
# 1. Build
make clean && make

# 2. Run basic test
./alltoallv_sm

# 3. Run with debug
NCCL_DEBUG=INFO ./alltoallv_sm

# 4. Check results
grep -i "verified\|error\|fail" test_output.log
```

## 🔍 What to Look For

### ✅ Success Indicators
- "AlltoAllV completed successfully"
- "Results verified correctly"
- "All resources cleaned up successfully"

### ❌ Error Indicators
- "ERROR" or "WARN" messages
- "Verification failed"
- Segmentation fault
- "Symmetric memory not supported"

## 🐛 Common Fixes

| Problem | Solution |
|---------|----------|
| Build fails | `cd ../../.. && make` (build NCCL first) |
| No GPUs found | Check `nvidia-smi` |
| CE not used | Set `NCCL_CTA_POLICY=0` |
| Verification fails | Check sendcounts/recvcounts arrays |

## 📊 Debug Commands

```bash
# Full debug output
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm 2>&1 | tee debug.log

# Check which backend is used
NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | grep -i "ce\|backend"

# Test with 2 GPUs only
NTHREADS=2 ./alltoallv_sm

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 ./alltoallv_sm
```

## 📝 Expected Output

```
Starting AlltoAllV example with 2 ranks
  Rank 0 communicator initialized using device 0
  Rank 1 communicator initialized using device 1
Setting up AlltoAllV with variable-sized transfers
  Send counts per rank: 1024 512
Symmetric Memory allocation
Starting AlltoAllV with variable-sized transfers
AlltoAllV completed successfully
Rank 0 verification:
  Received from rank 0: 1024 elements at offset 0 - OK
  Received from rank 1: 512 elements at offset 1024 - OK
Rank 0: Results verified correctly
All resources cleaned up successfully
```

For detailed debugging, see [DEBUG_GUIDE.md](DEBUG_GUIDE.md)
