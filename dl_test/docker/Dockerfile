# Multi-stage build for optimization
FROM python:3.10.6-slim AS builder

# Install build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    build-essential \
    wget \
    git \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies file
COPY requirements.txt .

# Upgrade pip before installing packages
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version (saves space)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Remove mmcv packages and install requirements.txt
RUN pip uninstall -y mmcv-lite mmcv mmengine mmdet mmpose || true
RUN pip install --no-cache-dir -r requirements.txt

# Compile and install mmcv from source
RUN pip install --no-cache-dir --no-binary mmcv mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# Install OpenMIM and all packages efficiently
RUN pip install openmim
RUN mim install mmengine==0.10.7 mmdet==3.2.0 mmpose==1.3.2

# Create required directory structure for mmpose
RUN mkdir -p /usr/local/lib/python3.10/site-packages/mmpose/.mim/checkpoints

# Download only AP-10K checkpoint file
RUN wget -O /usr/local/lib/python3.10/site-packages/mmpose/.mim/checkpoints/hrnet_w32_ap10k_256x256-18aac840_20211029.pth \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth

# Clean build cache
RUN pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# === Final runtime image ===
FROM python:3.10.6-slim

# Install runtime libraries only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary application files
COPY backend/ ./backend/
COPY training/ ./training/
COPY docker/.env ./

# Create directories
RUN mkdir -p uploads output_keypoints

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Application execution command
CMD ["python", "backend/main.py"]
