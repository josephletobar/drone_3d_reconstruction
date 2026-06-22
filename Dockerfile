FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    qt5-qmake \
    libqglviewer-dev-qt5 \
    libcgal-dev \
    libceres-dev \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install tqdm

# Build COLMAP from source
WORKDIR /tmp/colmap
RUN git clone https://github.com/colmap/colmap.git . && \
    mkdir build && \
    cd build && \
    cmake .. -DTESTS_ENABLED=OFF -DCUDA_ENABLED=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/colmap

# Create app directory
WORKDIR /app
COPY orchestrate.py /app/orchestrate.py

ENTRYPOINT ["python3", "/app/orchestrate.py"]
