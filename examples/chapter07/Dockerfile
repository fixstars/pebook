FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base

RUN \
  --mount=type=cache,target=/var/lib/apt/lists \
  --mount=type=cache,target=/var/cache/apt/archives \
  apt-get update && apt-get upgrade -y && \
  apt-get install wget -yq && \
  apt-get install -y \
     build-essential g++ gcc \
     libgl1-mesa-dev libglib2.0-0 \
     openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git\
     cmake ninja-build
# nsys
RUN \
  --mount=type=cache,target=/var/lib/apt/lists \
  --mount=type=cache,target=/var/cache/apt/archives \
  bash -c 'echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list' && \
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
  apt-get update && \
  apt-get install -y nsight-systems-cli


FROM base AS main

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"
RUN \
  --mount=type=cache,target=/var/lib/apt/lists \
  --mount=type=cache,target=/var/cache/apt/archives \
  apt-get update && apt-get install -y pipx sudo

# H100
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS="1"
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX 9.0+PTX"

ARG UID=0
ARG GID=0
ARG USERNAME=root
RUN  \
  --mount=type=cache,target=/var/lib/apt/lists \
  --mount=type=cache,target=/var/cache/apt/archives \
  if ! getent passwd $USERNAME > /dev/null 2>&1; then [ "$UID" -ne 0 -a "$GID" -ne 0 ] && groupadd --gid $GID $USERNAME && /usr/sbin/useradd -m --uid $UID --gid $GID -m $USERNAME -s /bin/bash && gpasswd -a $USERNAME sudo >/dev/null && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers; fi
USER $USERNAME
ARG HOME=/home/$USERNAME
ENV HOME=$HOME
RUN mkdir -p $HOME


# uv
WORKDIR /work
ENV PATH=$HOME/.local/bin/:$PATH
RUN --mount=type=cache,target=$HOME/.cache/pip pipx install uv
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .
RUN uv venv
RUN echo "source /work/.venv/bin/activate" >>$HOME/.bashrc
RUN uv run uv sync
RUN uv run pipx install openmim==0.3.9
RUN uv run pipx install ninja==1.11.1.4

# 互換性
# https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/notes/faq.md
# MMDetection3D version  MMEngine version	     MMCV version	          MMDetection version
# main	               mmengine>=0.8.0, <1.0.0	mmcv>=2.0.0rc4, <2.2.0	mmdet>=3.0.0rc5, <3.4.0
# v1.4.0	               mmengine>=0.8.0, <1.0.0	mmcv>=2.0.0rc4, <2.2.0	mmdet>=3.0.0rc5, <3.4.0
# v1.3.0	               mmengine>=0.8.0, <1.0.0	mmcv>=2.0.0rc4, <2.2.0	mmdet>=3.0.0rc5, <3.3.0
# v1.2.0	               mmengine>=0.8.0, <1.0.0	mmcv>=2.0.0rc4, <2.1.0	mmdet>=3.0.0, <3.2.0
# v1.1.1	               mmengine>=0.7.1, <1.0.0	mmcv>=2.0.0rc4, <2.1.0	mmdet>=3.0.0, <3.1.0

#install mmdet
RUN uv run python -m mim install mmengine==0.10.6 'numpy<2'
# # RUN git clone https://github.com/open-mmlab/mmengine.git -b v0.10.6 /work/mmdet/mmengine
# # WORKDIR /mmdet/mmengine
# # RUN --mount=type=cache,mode=0755,target=$HOME/.cache/pip pip install -e . -v

RUN uv run python -m mim install mmcv==2.1.0 'numpy<2'
# RUN git clone https://github.com/open-mmlab/mmcv.git -b v2.1.0 /work/mmdet/mmcv
# WORKDIR /work/mmdet/mmcv
# RUN bash -c "source /work/.venv/bin/activate && uv run --active setup.py develop"

RUN uv run python -m mim install mmdet==3.3.0 'numpy<2'
# # RUN git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 /work/mmdet/mmdetection
# # WORKDIR /work/mmdet/mmdetection
# # RUN --mount=type=cache,mode=0755,target=$HOME/.cache/pip pip install -e . -v


# # mmdetection3d
# RUN uv run python -m mim install mmdet3d==1.4.0 'numpy<2' 'matplotlib<3.6.0'
