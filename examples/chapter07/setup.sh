#!/bin/bash

# BEVFusion への高速化コードの適用と build とインストール

cd `dirname $0`
source /work/.venv/bin/activate
cd mmdetection3d

# numpy のアプデに対応
sed -i 's/np.long/np.int64/' mmdet3d/datasets/transforms/dbsampler.py
# H100 settings
sed -i 's/gencode=arch=compute_70,code=sm_70/gencode=arch=compute_90,code=sm_90/' projects/BEVFusion/setup.py
sed -i 's/gencode=arch=compute_75,code=sm_75/gencode=arch=compute_90,code=compute_90/' projects/BEVFusion/setup.py
# apply PE code
cp ../pe_BEVFusion/bevfusion/ops/src/pe_voxelization_cuda.cu projects/BEVFusion/bevfusion/ops/voxel/src/pe_voxelization_cuda.cu
sed -i 's/ hard_voxelize_gpu(/ hard_voxelize_gpu_pe(/' projects/BEVFusion/bevfusion/ops/voxel/src/voxelization.h
sed -i 's/src\/voxelization_cuda.cu/src\/pe_voxelization_cuda.cu/' projects/BEVFusion/setup.py
# install BEVFusion
bash -c "source /work/.venv/bin/activate && uv run --active projects/BEVFusion/setup.py develop"
