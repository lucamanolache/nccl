#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Default to the in-tree NCCL build if NCCL_HOME is not provided. This makes the
# examples pick up the freshly built headers/libs without requiring the user to
# export NCCL_HOME manually.
ifeq ($(strip $(NCCL_HOME)),)
NCCL_HOME := $(abspath $(CURDIR)/../../..)/build
endif

# Make sure NCCL headers are found and libraries are linked
ifneq ($(strip $(NCCL_HOME)),)
CXXFLAGS := -I$(NCCL_HOME)/include $(CXXFLAGS)
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

# Build configuration
INCLUDES = -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include
LIBRARIES = -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64
LDFLAGS = -lcudart -lnccl -Wl,-rpath,$(NCCL_HOME)/lib


# MPI configuration
ifeq ($(MPI), 1)

ifdef MPI_HOME
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIRUN ?= $(MPI_HOME)/bin/mpirun
else
MPICXX ?= mpicxx
MPIRUN ?= mpirun
endif

CXXFLAGS += -DMPI_SUPPORT
endif
