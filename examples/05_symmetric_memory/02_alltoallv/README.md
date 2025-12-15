# NCCL Symmetric Memory AlltoAllV Example

This example demonstrates how to use NCCL's symmetric memory feature for AlltoAllV (All-to-All Variable) collective operations.

## Overview

AlltoAllV allows each rank to send different amounts of data to every other rank. This is useful when:
- Data importance varies by destination rank
- Bandwidth allocation needs to be prioritized
- Variable-sized messages need to be exchanged

## Key Concepts

1. **Variable-Sized Transfers**: Each rank can send different amounts of data to different destination ranks
2. **Send Buffer Organization**: Data is ordered by importance - most important data (for rank 0) comes first
3. **Symmetric Memory**: All ranks register symmetric memory windows for optimal performance
4. **Send/Receive Counts**: Arrays specify how many elements to send/receive from each rank

## Building

```bash
cd examples/05_symmetric_memory/02_alltoallv
make
```

## Running

### With pthreads (default):
```bash
./alltoallv_sm
```

### With MPI:
```bash
make MPI=1
mpirun -np 4 ./alltoallv_sm
```

## Test Pattern

The example demonstrates:
- **Send counts**: Each rank sends more data to lower-ranked destinations (higher priority)
- **Data pattern**: Each element contains source rank, destination rank, and element index
- **Verification**: Checks that data is received in the correct positions

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
Verification:
  Received from rank 0: X elements at offset Y
  ...
Results structure verified correctly
```

## Notes

- This example assumes AlltoAllV API is available. If not, you may need to use internal APIs or implement a wrapper.
- The send buffer is organized with data for rank 0 first, then rank 1, etc., matching the importance ordering.
- All ranks must register symmetric memory windows before the collective operation.
