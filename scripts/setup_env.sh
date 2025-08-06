#!/bin/bash

# Iron Man Suit Simulation - Environment Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Iron Man Suit Simulation Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ is required but not installed"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is required but not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r backend/requirements.txt
    print_success "Python dependencies installed"
}

# Install system dependencies (macOS)
install_system_deps_macos() {
    print_status "Installing system dependencies (macOS)..."
    if command -v brew &> /dev/null; then
        brew install cmake gcc make
        print_success "System dependencies installed via Homebrew"
    else
        print_warning "Homebrew not found. Please install cmake, gcc, and make manually"
    fi
}

# Install system dependencies (Ubuntu/Debian)
install_system_deps_ubuntu() {
    print_status "Installing system dependencies (Ubuntu/Debian)..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git
    print_success "System dependencies installed"
}

# Build C/C++ components
build_cpp_components() {
    print_status "Building C/C++ components..."
    cd backend/aerodynamics/physics_plugin
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake -DENABLE_TESTING=ON ..
    make -j$(nproc)
    
    print_success "C/C++ components built successfully"
    cd ../../../..
}

# Run tests
run_tests() {
    print_status "Running tests..."
    cd backend
    
    # Run Python tests
    python -m pytest aerodynamics/tests/ -v
    
    # Run C++ tests
    cd aerodynamics/physics_plugin/build
    ctest --output-on-failure
    cd ../../../..
    
    print_success "Tests completed"
}

# Create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Create default simulation config
    cat > backend/aerodynamics/simulations/configs/suit_hover.yaml << EOF
# Iron Man Suit Hover Configuration
suit:
  mass: 100.0  # kg
  wing_area: 2.0  # m^2
  Cl0: 0.1
  Cld_alpha: 5.0
  Cd0: 0.02
  k: 0.1

simulation:
  dt: 0.01  # seconds
  duration: 10.0  # seconds
  initial_state: [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, vx, vy, vz]

control:
  target_altitude: 150.0  # meters
  max_thrust: 2000.0  # Newtons
  kp: 1.0
  ki: 0.1
  kd: 0.5
EOF
    
    print_success "Configuration files created"
}

# Main setup function
main() {
    print_status "Starting Iron Man Suit Simulation setup..."
    
    # Check prerequisites
    check_python
    check_pip
    
    # Detect OS and install system dependencies
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_system_deps_macos
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        install_system_deps_ubuntu
    else
        print_warning "Unsupported OS. Please install cmake, gcc, and make manually"
    fi
    
    # Setup Python environment
    create_venv
    activate_venv
    install_python_deps
    
    # Build components
    build_cpp_components
    
    # Create configs
    create_configs
    
    # Run tests
    run_tests
    
    print_success "🎉 Iron Man Suit Simulation environment setup complete!"
    print_status "To activate the environment, run: source venv/bin/activate"
    print_status "To run a simulation: python backend/aerodynamics/simulations/run_simulation.py --plot"
}

# Run main function
main "$@"


