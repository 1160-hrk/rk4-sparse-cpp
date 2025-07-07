FROM ubuntu:22.04

# 基本的なビルドツールとコンパイラのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libeigen3-dev \
    pybind11-dev \
    # 高性能BLASライブラリ
    libopenblas-dev \
    liblapack-dev \
    # プロファイリングツール
    linux-tools-generic \
    valgrind \
    # より新しいコンパイラ
    g++-12 \
    clang-14 \
    # その他の開発ツール
    gdb \
    ninja-build \
    ccache \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create the user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 環境変数の設定
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12
ENV CFLAGS="-O3 -march=native -ffast-math"
ENV CXXFLAGS="-O3 -march=native -ffast-math"
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /workspace
RUN chown $USERNAME:$USERNAME /workspace

# Copy requirements files
COPY --chown=$USERNAME:$USERNAME requirements.txt requirements_dev.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt && \
    pip install -r requirements_dev.txt && \
    # 追加のPythonパッケージ
    pip install \
    line_profiler \
    memory_profiler \
    pytest-benchmark

RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the default user
USER $USERNAME
