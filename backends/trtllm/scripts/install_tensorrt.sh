#!/bin/bash

set -ex

TRT_VER_BASE="10.11.0"
TRT_VER_FULL="${TRT_VER_BASE}.33"
CUDA_VER="12.8"
CUDNN_VER="9.8.0.87-1"
NCCL_VER="2.25.1-1+cuda${CUDA_VER}"
CUBLAS_VER="${CUDA_VER}.4.1-1"
NVRTC_VER="${CUDA_VER}.93-1"

for i in "$@"; do
    case $i in
        --TRT_VER=?*) TRT_VER="${i#*=}";;
        --CUDA_VER=?*) CUDA_VER="${i#*=}";;
        --CUDNN_VER=?*) CUDNN_VER="${i#*=}";;
        --NCCL_VER=?*) NCCL_VER="${i#*=}";;
        --CUBLAS_VER=?*) CUBLAS_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

NVCC_VERSION_OUTPUT=$(nvcc --version)
if [[ $(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1) != ${CUDA_VER} ]]; then
  echo "The version of pre-installed CUDA is not equal to ${CUDA_VER}."
  exit 1
fi

install_ubuntu_requirements() {
    apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    rm /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list

    apt-get update
    if [[ $(apt list --installed | grep libcudnn9) ]]; then
      apt-get remove --purge -y --allow-change-held-packages libcudnn9*
    fi
    if [[ $(apt list --installed | grep libnccl) ]]; then
      apt-get remove --purge -y --allow-change-held-packages libnccl*
    fi
    if [[ $(apt list --installed | grep libcublas) ]]; then
      apt-get remove --purge -y --allow-change-held-packages libcublas*
    fi
    if [[ $(apt list --installed | grep cuda-nvrtc-dev) ]]; then
      apt-get remove --purge -y --allow-change-held-packages cuda-nvrtc-dev*
    fi

    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')

    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-12=${CUDNN_VER} \
        libcudnn9-dev-cuda-12=${CUDNN_VER} \
        libnccl2=${NCCL_VER} \
        libnccl-dev=${NCCL_VER} \
        libcublas-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        libcublas-dev-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER}

    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

install_centos_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    yum -y update
    yum -y install epel-release
    yum remove -y libnccl* && yum -y install libnccl-${NCCL_VER} libnccl-devel-${NCCL_VER}
    yum remove -y libcublas* && yum -y install libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER} libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}
    yum clean all
}

install_tensorrt() {
    #PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    #PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

    TRT_CUDA_VERSION="12.9"

    if [ -z "$RELEASE_URL_TRT" ];then
        ARCH=${TRT_TARGETARCH}
        if [ -z "$ARCH" ];then ARCH=$(uname -m);fi
        if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi
        if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
        RELEASE_URL_TRT="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_VER_BASE}/tars/TensorRT-${TRT_VER_FULL}.Linux.${ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"
    fi

    wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
    tar -xf /tmp/TensorRT.tar -C /usr/local/
    mv /usr/local/TensorRT-${TRT_VER_FULL} /usr/local/tensorrt
    # pip3 install --no-cache-dir /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
    rm -rf /tmp/TensorRT.tar
    #echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> "${ENV}"

    # We only run inference, so the nvinfer_builder_resource libs are not required.
    # We do not need static libraries either.
    rm -f /usr/local/tensorrt/lib/libnvinfer_vc_plugin_static.a \
          /usr/local/tensorrt/lib/libnvinfer_plugin_static.a \
          /usr/local/tensorrt/lib/libnvinfer_static.a \
          /usr/local/tensorrt/lib/libnvinfer_dispatch_static.a \
          /usr/local/tensorrt/lib/libnvinfer_lean_static.a \
          /usr/local/tensorrt/lib/libnvonnxparser_static.a \
          /usr/local/tensorrt/lib/libonnx_proto.a \
          /usr/local/tensorrt/lib/libnvinfer_lean.so* \
          /usr/local/tensorrt/lib/libnvinfer_builder_resource*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  debian)
    install_ubuntu_requirements
    install_tensorrt
    ;;
  ubuntu)
    #install_ubuntu_requirements
    install_tensorrt
    ;;
  centos)
    install_centos_requirements
    install_tensorrt
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
