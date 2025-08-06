#!/bin/bash
# Script to enable OpenMP support on macOS

echo "==== OpenMP Setup for macOS ===="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh"
    exit 1
fi

# Install libomp
echo "Installing libomp via Homebrew..."
brew install libomp

# Get libomp paths
LIBOMP_PREFIX=$(brew --prefix libomp)

echo ""
echo "libomp installed at: $LIBOMP_PREFIX"
echo ""
echo "To build with OpenMP support, use these CMake flags:"
echo ""
echo "cmake \\"
echo "  -DOpenMP_C_FLAGS=\"-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include\" \\"
echo "  -DOpenMP_C_LIB_NAMES=\"omp\" \\"
echo "  -DOpenMP_omp_LIBRARY=\"${LIBOMP_PREFIX}/lib/libomp.dylib\" \\"
echo "  -DENABLE_TESTING=ON \\"
echo "  .."
echo ""
echo "Or add these to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "export OpenMP_C_FLAGS=\"-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include\""
echo "export OpenMP_C_LIB_NAMES=\"omp\""
echo "export OpenMP_omp_LIBRARY=\"${LIBOMP_PREFIX}/lib/libomp.dylib\""
echo ""
echo "Then build normally:"
echo "cmake -DENABLE_TESTING=ON .."
echo "cmake --build . -- -j4"