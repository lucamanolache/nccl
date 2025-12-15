/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * NCCL Symmetric Memory AlltoAllV Example
 *
 * This example demonstrates how to use NCCL's symmetric memory feature
 * for AlltoAllV (All-to-All Variable) collective operations. AlltoAllV
 * allows each rank to send different amounts of data to every other rank.
 *
 * Learning Objectives:
 * - Learn how to use AlltoAllV with symmetric memory windows
 * - Understand variable-sized data transfers in collectives
 * - See how send buffers are organized by importance/priority
 *
 */

/*
 * This function can be called inside an MPI rank or pthread thread. The
 * initialization and broadcast are implemented in common/src/utils.cc for
 * easier readability. For fully integrated examples using pthreads or MPI see
 * examples in 01_communicators.
 */
void *alltoallV(int my_rank, int total_ranks, int local_device,
                 int devices_per_rank) {

  // ========================================================================
  // STEP 1: Initialize NCCL Communicator and Setup
  // ========================================================================

  ncclUniqueId nccl_unique_id;
  if (my_rank == 0) {
    printf("Starting AlltoAllV example with %d ranks\n", total_ranks);
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID.
  // This step ensures all ranks have the same unique ID for communicator
  // creation
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  // Each rank manages its assigned GPU device
  CUDACHECK(cudaSetDevice(local_device));

  // Initialize NCCL communicator
  // This creates the communication context for collective operations
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d communicator initialized using device %d\n", my_rank,
         local_device);

  // ========================================================================
  // STEP 2: Setup Variable-Sized Data Configuration
  // ========================================================================

  if (my_rank == 0) {
    printf("Setting up AlltoAllV with variable-sized transfers\n");
  }

  // Define send counts: each rank sends different amounts to different ranks
  // Data is ordered by importance - most important data first
  // Example: Rank 0 sends more data to rank 0, less to rank 1, etc.
  size_t *sendcounts = (size_t *)malloc(total_ranks * sizeof(size_t));
  size_t *recvcounts = (size_t *)malloc(total_ranks * sizeof(size_t));
  size_t *sdispls = (size_t *)malloc(total_ranks * sizeof(size_t));
  size_t *rdispls = (size_t *)malloc(total_ranks * sizeof(size_t));

  // Calculate send counts: send more data to lower ranks (higher priority)
  // This simulates sending more important data first
  size_t max_elements_per_rank = 1024;
  size_t total_send_elements = 0;
  size_t total_recv_elements = 0;

  for (int r = 0; r < total_ranks; r++) {
    // Send more data to ranks with lower indices (higher priority)
    // Formula: more data to rank 0, less to higher ranks
    sendcounts[r] = max_elements_per_rank / (r + 1);
    if (sendcounts[r] == 0)
      sendcounts[r] = 1; // Ensure at least 1 element
    total_send_elements += sendcounts[r];

    // For symmetric alltoallv, recvcounts match sendcounts pattern
    // Rank r receives sendcounts[r] elements from each source rank
    recvcounts[r] = sendcounts[r];
    total_recv_elements += recvcounts[r];
  }

  // Calculate displacements
  sdispls[0] = 0;
  for (int r = 1; r < total_ranks; r++) {
    sdispls[r] = sdispls[r - 1] + sendcounts[r - 1];
  }

  rdispls[0] = 0;
  for (int r = 1; r < total_ranks; r++) {
    rdispls[r] = rdispls[r - 1] + recvcounts[r - 1];
  }

  if (my_rank == 0) {
    printf("  Send counts per rank: ");
    for (int r = 0; r < total_ranks; r++) {
      printf("%zu ", sendcounts[r]);
    }
    printf("\n");
    printf("  Total send elements: %zu, Total recv elements: %zu\n",
           total_send_elements, total_recv_elements);
  }

  // ========================================================================
  // STEP 3: Allocate Memory Using NCCL Allocator
  // ========================================================================

  if (my_rank == 0) {
    printf("Symmetric Memory allocation\n");
  }

  size_t element_size = sizeof(float);
  size_t send_size_bytes = total_send_elements * element_size;
  size_t recv_size_bytes = total_recv_elements * element_size;

  printf("  Rank %d allocating send: %.2f MB, recv: %.2f MB\n", my_rank,
         (float)send_size_bytes / (1024 * 1024),
         (float)recv_size_bytes / (1024 * 1024));

  float *h_send_data = (float *)malloc(send_size_bytes);
  float *h_recv_data = (float *)malloc(recv_size_bytes);

  // Allocate buffers using NCCL allocator
  // NCCL's allocator is compatible with symmetric memory layouts
  void *d_sendbuff;
  void *d_recvbuff;
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, send_size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, recv_size_bytes));

  // ========================================================================
  // STEP 4: Register Symmetric Memory Windows
  // ========================================================================

  /* Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric
   * buffers among all ranks in collectives.
   * Every rank needs to call ncclCommWindowRegister to register its buffers.
   */

  // Register symmetric memory windows with NCCL
  ncclWindow_t send_win;
  ncclWindow_t recv_win;
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, send_size_bytes, &send_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, recv_size_bytes, &recv_win,
                                   NCCL_WIN_COLL_SYMMETRIC));

  // ========================================================================
  // STEP 5: Initialize Data and Prepare for Communication
  // ========================================================================

  // Initialize send data - ordered by importance
  // Most important data (for rank 0) comes first, then rank 1, etc.
  size_t offset = 0;
  for (int r = 0; r < total_ranks; r++) {
    // Fill data for rank r with pattern: rank value + destination rank
    // This allows verification later
    for (size_t i = 0; i < sendcounts[r]; i++) {
      h_send_data[offset + i] = (float)(my_rank * 1000 + r * 100 + i);
    }
    offset += sendcounts[r];
  }

  // Initialize receive buffer to zero for verification
  memset(h_recv_data, 0, recv_size_bytes);

  CUDACHECK(cudaMemcpy(d_sendbuff, h_send_data, send_size_bytes,
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_recvbuff, h_recv_data, recv_size_bytes,
                       cudaMemcpyHostToDevice));

  printf("  Rank %d data initialized\n", my_rank);

  // Create stream for asynchronous operations
  // Streams allow overlapping computation and communication
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // ========================================================================
  // STEP 6: Perform AlltoAllV Operation
  // ========================================================================

  if (my_rank == 0) {
    printf("Starting AlltoAllV with variable-sized transfers\n");
  }

  // Perform AlltoAllV operation
  // Since symmetric memory is registered, NCCL can apply optimized algorithms
  NCCLCHECK(ncclAlltoAllV(d_sendbuff, sendcounts, sdispls,
                          d_recvbuff, recvcounts, rdispls,
                          ncclFloat, comm, stream));

  if (my_rank == 0) {
    printf("AlltoAllV completed successfully\n");
  }

  // ========================================================================
  // STEP 7: Verify Results and Validate Correctness
  // ========================================================================

  // Synchronize to ensure completion
  CUDACHECK(cudaStreamSynchronize(stream));

  // Copy results back for verification
  CUDACHECK(cudaMemcpy(h_recv_data, d_recvbuff, recv_size_bytes,
                       cudaMemcpyDeviceToHost));

  // Verify results
  // Each rank should receive data from all other ranks
  // Data from rank src should be at position src in receive buffer
  bool all_ok = true;
  printf("Rank %d verification:\n", my_rank);
  
  // Expected: Rank i receives recvcounts[i] elements from each source rank
  // For symmetric alltoallv: recvcounts[i] = sendcounts[i]
  for (int src = 0; src < total_ranks && all_ok; src++) {
    size_t recv_offset = rdispls[src];
    size_t expected_count = recvcounts[src];
    
    if (expected_count > 0) {
      // Verify pattern: data from rank src should have pattern (src * 1000 + my_rank * 100 + i)
      for (size_t i = 0; i < expected_count; i++) {
        float expected = (float)(src * 1000 + my_rank * 100 + i);
        float actual = h_recv_data[recv_offset + i];
        if (fabsf(actual - expected) > 0.001) {
          printf("  Rank %d: Verification failed at src=%d, index %zu: Expected %.1f, Got %.1f\n",
                 my_rank, src, i, expected, actual);
          all_ok = false;
          break;
        }
      }
      if (all_ok && my_rank == 0) {
        printf("  Received from rank %d: %zu elements at offset %zu - OK\n", src,
               expected_count, recv_offset);
      }
    }
  }

  if (all_ok) {
    printf("Rank %d: Results verified correctly\n", my_rank);
  } else {
    printf("Rank %d: Results verification failed\n", my_rank);
  }

  // ========================================================================
  // STEP 8: Cleanup and Resource Management
  // ========================================================================

  // Important: Cleanup must happen in the correct order
  // 1. Free host memory
  // 2. Deregister symmetric memory windows
  // 3. Free device memory
  // 4. Destroy CUDA resources
  // 5. Finalize and destroy NCCL communicator

  free(h_send_data);
  free(h_recv_data);
  free(sendcounts);
  free(recvcounts);
  free(sdispls);
  free(rdispls);

  // Deregister symmetric memory windows from communicator
  // This must happen before freeing the buffers or destroying the
  // communicator
  NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
  NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));
  printf("  Rank %d symmetric memory windows deregistered\n", my_rank);

  // Free device memory allocated by NCCL
  NCCLCHECK(ncclMemFree(d_sendbuff));
  NCCLCHECK(ncclMemFree(d_recvbuff));

  // Destroy CUDA stream
  CUDACHECK(cudaStreamDestroy(stream));

  // Finalize and destroy NCCL communicator
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  if (my_rank == 0) {
    printf("All resources cleaned up successfully\n");
    printf("Example completed - demonstrated AlltoAllV with symmetric memory\n");
  }

  return NULL;
}

int main(int argc, char *argv[]) {
  // Run example using the standard test framework
  // This handles MPI/pthread initialization, device assignment, and cleanup
  return run_example(argc, argv, alltoallV);
}
