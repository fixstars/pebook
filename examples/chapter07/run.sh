#!/bin/bash

cd `dirname $0`

source /work/.venv/bin/activate
cd mmdetection3d
# uv run --active bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8 >log.txt 2>&1

LIDAR_PRETRAINED_CHECKPOINT=work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth
IMAGE_PRETRAINED_BACKBONE=swint-nuimages-pretrained.pth
CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# uv run --active bash tools/dist_train.sh ${CONFIG} 8 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}

cd ../
LIDAR_PRETRAINED_CHECKPOINT=mmdetection3d/$LIDAR_PRETRAINED_CHECKPOINT
IMAGE_PRETRAINED_BACKBONE=mmdetection3d/$IMAGE_PRETRAINED_BACKBONE
CONFIG=pe_BEVFusion/configs/bevfusion_lidar-cam_pe.py
PROFILE=""
# PROFILE=" nsys profile --gpu-metrics-devices all --capture-range-end=stop  --capture-range=cudaProfilerApi --trace-fork-before-exec=true"
./setup.sh
PYTHONPATH="$(pwd):$PYTHONPATH"
sudo -E /home/root/.local/bin/uv run --active ${PROFILE} bash -e mmdetection3d/tools/dist_train.sh ${CONFIG} 8 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}

