FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    software-properties-common \
    git \
    libmysqlclient-dev \
    pkg-config \
    libperl-dev \
    libgtk-3-dev \
    libnotify-dev \
    libsdl2-mixer-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-2.0-0 \
    libvips \
    python3-pip \
    python3-opencv \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3.11 -m pip install --upgrade pip setuptools wheel packaging

RUN python3.11 -m pip install .

RUN python3.11 -m pip install numpy==1.26.4 psutil

RUN python3.11 -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3.11 -m pip install transformers==4.51.0 accelerate==1.2.1

RUN python3.11 -m pip install flash-attn==2.7.4.post1 \
    --no-build-isolation

RUN python3.11 -m pip install -U timm httpx

ENV DISPLAY=:0.0
ENV DLClight=True

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

EXPOSE 6000
EXPOSE 7000

