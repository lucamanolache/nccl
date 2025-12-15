# Step-by-Step Debugging Guide for AlltoAllV

## Prerequisites Check

First, verify your environment:

```bash
# Check CUDA is available
nvcc --version
nvidia-smi

# Check NCCL is built
cd /path/to/nccl
ls -la build/lib/libnccl.so*  # or wherever NCCL is installed

# Check you have multiple GPUs (or can use MPI)
nvidia-smi | grep -c "GPU"
```

## Step 1: Build the Test

```bash
# Navigate to test directory
cd examples/05_symmetric_memory/02_alltoallv

# Clean any previous builds
make clean

# Build the test
make

# Check if build succeeded
ls -la alltoallv_sm
```

**If build fails:**
- Check that NCCL is built: `cd ../../.. && make`
- Check CUDA paths: `echo $CUDA_HOME`
- Check compiler: `which g++` or `which nvcc`

## Step 2: Run Basic Test

```bash
# Run with minimal output first
./alltoallv_sm
```

**Expected output:**
```
Starting AlltoAllV example with X ranks
  Rank 0 communicator initialized using device 0
  Rank 1 communicator initialized using device 1
...
```

**If it fails immediately:**
- Check GPUs: `nvidia-smi`
- Check permissions: `ls -la /dev/nvidia*`
- Try with fewer GPUs: `NTHREADS=2 ./alltoallv_sm`

## Step 3: Run with Debug Output

```bash
# Enable NCCL debug logging
NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | tee debug.log
```

**Look for:**
- ✅ "Init CE" - CE initialization successful
- ✅ "symmetric memory windows registered" - Windows setup OK
- ✅ "AlltoAllV completed successfully" - Operation succeeded
- ❌ Any ERROR or WARN messages

## Step 4: Verify CE Backend is Used

```bash
# Force CE backend and see detailed output
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm 2>&1 | grep -i "ce\|alltoallv\|symmetric"
```

**Expected:**
- Messages about CE initialization
- Symmetric memory registration
- AlltoAllV operation

## Step 5: Check Data Verification

The test should print verification results. Look for:

```
Rank X verification:
  Received from rank 0: X elements at offset Y - OK
  ...
Rank X: Results verified correctly
```

**If verification fails:**
- Check the error message - it will show expected vs actual values
- Verify sendcounts/recvcounts arrays are correct
- Check that displacements are calculated properly

## Step 6: Test with Different Configurations

### Test with 2 GPUs only
```bash
NTHREADS=2 ./alltoallv_sm
```

### Test with specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 ./alltoallv_sm
```

### Test with MPI (if available)
```bash
make MPI=1
mpirun -np 2 ./alltoallv_sm
```

## Common Issues and Solutions

### Issue 1: "Symmetric memory not supported"

**Symptoms:**
```
ERROR: Symmetric memory windows not found
```

**Solutions:**
- Ensure CUDA 12.5+ (check: `nvidia-smi`)
- Verify buffers allocated with `ncclMemAlloc` (already done in test)
- Check windows registered with `NCCL_WIN_COLL_SYMMETRIC` flag

### Issue 2: "CE not implemented"

**Symptoms:**
```
WARN: CE not implemented for AlltoAllV
```

**Solutions:**
- Check CUDA driver version: `cat /proc/driver/nvidia/version`
- Set `NCCL_CTA_POLICY=0` explicitly
- Verify single-node setup (CE requires `nNodes == 1`)

### Issue 3: "Verification failed"

**Symptoms:**
```
Verification failed at index X: Expected Y, Got Z
```

**Debug steps:**
1. Check sendcounts array values
2. Verify send buffer initialization pattern
3. Check receive buffer layout matches expectations
4. Add debug prints to see actual values:

```cpp
// Add to main.cc after AlltoAllV call
printf("Rank %d: First few recv values: ", my_rank);
for (int i = 0; i < 10; i++) {
  printf("%.1f ", h_recv_data[i]);
}
printf("\n");
```

### Issue 4: Build errors

**Symptoms:**
```
error: 'ncclAlltoAllV' was not declared
```

**Solutions:**
- Ensure you've built NCCL with the new API: `cd ../../.. && make`
- Check `nccl.h` includes the function declaration
- Rebuild NCCL library if needed

### Issue 5: Segmentation fault

**Debug steps:**
```bash
# Run with gdb
gdb ./alltoallv_sm
(gdb) run
(gdb) bt  # when it crashes, get backtrace

# Or use valgrind
valgrind --tool=memcheck ./alltoallv_sm
```

## Step 7: Add Debug Prints (if needed)

If you need more visibility, add prints to `main.cc`:

```cpp
// After line 207 (AlltoAllV call)
printf("Rank %d: About to call ncclAlltoAllV\n", my_rank);
NCCLCHECK(ncclAlltoAllV(...));
printf("Rank %d: ncclAlltoAllV returned successfully\n", my_rank);

// Before verification
printf("Rank %d: Send counts: ", my_rank);
for (int i = 0; i < total_ranks; i++) {
  printf("%zu ", sendcounts[i]);
}
printf("\n");
```

## Step 8: Verify End-to-End

Run a complete test cycle:

```bash
# Clean build
make clean
make

# Run with full debug
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm 2>&1 | tee full_debug.log

# Check log for issues
grep -i "error\|warn\|fail" full_debug.log

# Verify success messages
grep -i "completed\|verified\|success" full_debug.log
```

## Success Criteria

✅ Build completes without errors  
✅ Program runs without crashes  
✅ All ranks initialize successfully  
✅ Symmetric memory windows register  
✅ AlltoAllV operation completes  
✅ Data verification passes  
✅ No ERROR or WARN messages (except expected ones)  

## Quick Debug Checklist

- [ ] NCCL library built and accessible
- [ ] CUDA 12.5+ available
- [ ] Multiple GPUs available (or MPI configured)
- [ ] Test builds successfully
- [ ] Test runs without immediate crash
- [ ] All ranks initialize
- [ ] Symmetric memory registers
- [ ] AlltoAllV completes
- [ ] Verification passes
- [ ] No unexpected errors

## Getting Help

If issues persist:

1. **Collect debug info:**
   ```bash
   NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | tee debug.log
   ```

2. **Check system info:**
   ```bash
   nvidia-smi
   nvcc --version
   cat /proc/driver/nvidia/version
   ```

3. **Verify NCCL build:**
   ```bash
   cd ../../..
   make clean
   make
   ```

4. **Test with simpler example:**
   ```bash
   cd ../01_allreduce
   make
   ./allreduce_sm  # Verify basic symmetric memory works
   ```
