FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential cmake python3 python3-pip python3-dev git wget \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pybind11 pytest

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
