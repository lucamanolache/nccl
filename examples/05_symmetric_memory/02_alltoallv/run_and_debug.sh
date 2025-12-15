#!/bin/bash
# Quick script to build, run, and debug AlltoAllV test

set -e  # Exit on error

echo "=========================================="
echo "AlltoAllV Test - Build and Debug Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "-----------------------------------"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Is CUDA installed?${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} CUDA found: $(nvcc --version | grep release)"

# Check GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -eq "0" ]; then
    echo -e "${RED}ERROR: No GPUs found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPU(s)"

# Check NCCL library
if [ -z "$NCCL_HOME" ]; then
    echo -e "${YELLOW}WARNING: NCCL_HOME not set. Assuming NCCL is in build path.${NC}"
else
    echo -e "${GREEN}✓${NC} NCCL_HOME: $NCCL_HOME"
fi

echo ""
echo "Step 2: Building test..."
echo "-----------------------------------"

# Clean previous build
make clean 2>/dev/null || true

# Build
if make; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${RED}✗${NC} Build failed"
    exit 1
fi

echo ""
echo "Step 3: Running basic test..."
echo "-----------------------------------"

# Run basic test
if ./alltoallv_sm 2>&1 | tee test_output.log; then
    echo -e "${GREEN}✓${NC} Test completed"
else
    echo -e "${RED}✗${NC} Test failed"
    echo ""
    echo "Last 20 lines of output:"
    tail -20 test_output.log
    exit 1
fi

echo ""
echo "Step 4: Checking for errors..."
echo "-----------------------------------"

# Check for errors in output
if grep -qi "error\|fail\|segmentation" test_output.log; then
    echo -e "${RED}✗${NC} Errors found in output:"
    grep -i "error\|fail\|segmentation" test_output.log | head -5
    exit 1
else
    echo -e "${GREEN}✓${NC} No errors found"
fi

# Check for success indicators
if grep -qi "Results verified correctly\|completed successfully" test_output.log; then
    echo -e "${GREEN}✓${NC} Success indicators found"
else
    echo -e "${YELLOW}⚠${NC} Success indicators not found - check output"
fi

echo ""
echo "Step 5: Running with debug output..."
echo "-----------------------------------"

# Run with debug
echo "Running with NCCL_DEBUG=INFO..."
NCCL_DEBUG=INFO ./alltoallv_sm 2>&1 | tee debug_output.log | tail -30

echo ""
echo "Step 6: Summary"
echo "-----------------------------------"
echo "Test output saved to: test_output.log"
echo "Debug output saved to: debug_output.log"
echo ""
echo "To view full debug output:"
echo "  cat debug_output.log"
echo ""
echo "To run with specific number of GPUs:"
echo "  NTHREADS=2 ./alltoallv_sm"
echo ""
echo "To run with CE backend:"
echo "  NCCL_DEBUG=INFO NCCL_CTA_POLICY=0 ./alltoallv_sm"
echo ""

if grep -qi "Results verified correctly" test_output.log; then
    echo -e "${GREEN}=========================================="
    echo "TEST PASSED!"
    echo "==========================================${NC}"
    exit 0
else
    echo -e "${YELLOW}=========================================="
    echo "TEST COMPLETED - VERIFY RESULTS MANUALLY"
    echo "==========================================${NC}"
    exit 0
fi
