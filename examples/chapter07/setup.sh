#!/bin/bash

cd `dirname $0`
# git submodule update -i
source /work/.venv/bin/activate
cd mmdetection3d
# install mmdet3d
bash -c "source /work/.venv/bin/activate && uv run --active setup.py build_ext && uv run --active setup.py develop"

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
