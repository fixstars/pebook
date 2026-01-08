#!/bin/bash

source /work/.venv/bin/activate
NGPU=8 # GPU枚数

# docker build時に行わなかった初期化を実行しておく
cd `dirname $0`
./setup.sh
ln -s ./mmdetection3d/data ./data

cd `dirname $0`

# 事前学習重みの作成
# 本書の解説では簡単のため高速化対象範囲外としています
cd mmdetection3d
LIDAR_PRETRAINED_CHECKPOINT=work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth
if [ ! -e $LIDAR_PRETRAINED_CHECKPOINT ]; then
  echo "Pretrained checkpoint not found. Run pretrain..."
  uv run --active bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${NGPU}
fi

# オリジナルコードの実行
LIDAR_PRETRAINED_CHECKPOINT=work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth
IMAGE_PRETRAINED_BACKBONE=swint-nuimages-pretrained.pth
CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# uv run --active bash tools/dist_train.sh ${CONFIG} 8 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}

# 高速化コードの実行
cd ../
LIDAR_PRETRAINED_CHECKPOINT=mmdetection3d/$LIDAR_PRETRAINED_CHECKPOINT
IMAGE_PRETRAINED_BACKBONE=mmdetection3d/$IMAGE_PRETRAINED_BACKBONE
CONFIG=pe_BEVFusion/configs/bevfusion_lidar-cam_pe.py
PROFILE=""
# PROFILE=" nsys profile --gpu-metrics-devices all --capture-range-end=stop  --capture-range=cudaProfilerApi --trace-fork-before-exec=true"
PYTHONPATH="$(pwd):$PYTHONPATH"
uv run --active ${PROFILE} bash -e mmdetection3d/tools/dist_train.sh ${CONFIG} ${NGPU} --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
