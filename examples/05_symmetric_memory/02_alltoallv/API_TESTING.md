# API Testing Guide: Public vs Internal APIs

## Overview

The AlltoAllV implementation has two API layers:

1. **Public API** (`ncclAlltoAllV`) - User-facing API that applications call
2. **Internal API** (`ncclCeAlltoAllV`) - CE backend implementation used internally

## Public API Testing

The public API is what users call in their applications. The test in `main.cc` uses this.

### Public API Signature
```c
ncclResult_t ncclAlltoAllV(
    const void* sendbuff,           // Send buffer
    const size_t* sendcounts,        // Array: elements to send to each rank
    const size_t* sdispls,           // Array: displacements in send buffer
    void* recvbuff,                  // Receive buffer
    const size_t* recvcounts,        // Array: elements to receive from each rank
    const size_t* rdispls,           // Array: displacements in receive buffer
    ncclDataType_t datatype,         // Data type (ncclFloat, etc.)
    ncclComm_t comm,                 // NCCL communicator
    cudaStream_t stream              // CUDA stream
);
```

### How to Test Public API

**Step 1: Build the test**
```bash
cd examples/05_symmetric_memory/02_alltoallv
make
```

**Step 2: Run the test**
```bash
./alltoallv_sm
```

The test automatically:
- Sets up symmetric memory windows
- Calls `ncclAlltoAllV()` public API
- Verifies results
- Uses CE backend if available (automatic)

### What the Public API Test Verifies

✅ API function call succeeds  
✅ Data is transferred correctly  
✅ Variable-sized transfers work  
✅ Symmetric memory integration  
✅ Multi-rank communication  

## Internal API (CE Backend)

The internal `ncclCeAlltoAllV` function is called automatically by the public API when:
- Symmetric memory windows are registered
- CE (CUDA Engine) is supported (CUDA 12.5+)
- `NCCL_CTA_POLICY=0` is set

### Internal API Signature
```c
ncclResult_t ncclCeAlltoAllV(
    struct ncclComm* comm,
    struct ncclCeCollArgs* args,  // Internal args structure
    cudaStream_t stream
);
```

### Testing Internal API Directly

**Note**: This is typically only done for internal development/debugging.

To verify the CE backend is being used:

```bash
# Enable debug output to see which backend is used
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm
```

Look for messages indicating CE collective execution.

### Force CE Backend Usage

The public API automatically routes to CE backend when conditions are met:
1. Symmetric memory windows registered (`NCCL_WIN_COLL_SYMMETRIC`)
2. Single node (`comm->nNodes == 1`)
3. CE policy enabled (`NCCL_CTA_POLICY=0`)
4. CUDA 12.5+ driver

## Testing Both APIs Together

The test in `main.cc` tests both layers:

1. **Public API Layer**: Your code calls `ncclAlltoAllV()`
2. **Internal Routing**: NCCL routes to appropriate backend (CE, kernel, etc.)
3. **CE Backend**: If conditions met, calls `ncclCeAlltoAllV()` internally
4. **Verification**: Results are checked regardless of backend used

### Test Flow

```
User Code
  ↓
ncclAlltoAllV() [Public API]
  ↓
ncclEnqueueCheck() → taskAppend() → ceCollTaskAppend()
  ↓
ncclCeAlltoAllV() [Internal CE API]
  ↓
ncclCeLaunchBatchOps() → CUDA operations
  ↓
Verification
```

## Running Tests

### Basic Test (Public API)
```bash
cd examples/05_symmetric_memory/02_alltoallv
make
./alltoallv_sm
```

### Test with Debug (See Internal Routing)
```bash
NCCL_DEBUG=INFO ./alltoallv_sm
```

### Test CE Backend Specifically
```bash
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm
```

### Test Multiple Scenarios
```bash
# Test with 4 GPUs
NTHREADS=4 ./alltoallv_sm

# Test with MPI
make MPI=1
mpirun -np 4 ./alltoallv_sm
```

## Verification

The test verifies:
- ✅ Public API call succeeds
- ✅ Data correctness (pattern matching)
- ✅ Variable sizes handled correctly
- ✅ All ranks receive correct data
- ✅ Symmetric memory works
- ✅ Backend selection (CE vs fallback)

## Debugging

### Check Which Backend is Used

```bash
NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | grep -i "ce\|backend\|collective"
```

### Verify CE Implementation

```bash
# Should show CE is implemented
NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm 2>&1 | grep "CE"
```

### Check Symmetric Memory

```bash
# Should show window registration
NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | grep -i "window\|symmetric"
```

## Summary

- **Public API**: Tested via `./alltoallv_sm` - this is what users call
- **Internal API**: Tested automatically when conditions are met
- **Both**: Work together transparently - public API routes to internal implementation
- **Verification**: Test validates end-to-end correctness regardless of backend

The test in `main.cc` exercises the full stack from public API through to CUDA operations.
