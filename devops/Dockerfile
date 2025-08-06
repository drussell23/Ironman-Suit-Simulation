# Iron Man Suit Simulation - Docker Container
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM ubuntu:20.04 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY backend/ ./backend/

# Build C/C++ components
RUN cd backend/aerodynamics/physics_plugin && \
    mkdir -p build && \
    cd build && \
    cmake -DENABLE_TESTING=OFF .. && \
    make -j$(nproc)

# Stage 2: Production environment
FROM ubuntu:20.04 as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libblas3 \
    liblapack3 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 ironman && \
    mkdir -p /app && \
    chown ironman:ironman /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/backend/aerodynamics/physics_plugin/build/libaerodynamics_physics_plugin.so ./backend/aerodynamics/physics_plugin/

# Copy scripts and configuration
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Set ownership
RUN chown -R ironman:ironman /app

# Switch to non-root user
USER ironman

# Expose port for API (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import numpy; print('OK')" || exit 1

# Default command
CMD ["python3", "backend/aerodynamics/simulations/run_simulation.py", "--plot"]


