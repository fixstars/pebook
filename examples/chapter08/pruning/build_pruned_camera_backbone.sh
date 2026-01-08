# configure the environment
. tool/environment.sh

export DEBUG_DATA=example-data
export DEBUG_MODEL=resnet50
export DEBUG_PRECISION=fp16

# build trt engine
trtexec --onnx=./model/resnet50/camera.backbone_fp16_w_sparse.onnx --fp16 --sparsity=enable \
    --inputIOFormats=fp16:chw,fp16:chw --outputIOFormats=fp16:chw,fp16:chw \
    --memPoolSize=workspace:2048 --saveEngine=./model/resnet50/build/camera.backbone.plan

# build app
mkdir -p build
cd build
cmake ..
make -j
cd ..

# execution and profiling
nsys profile -o prof_pruned  --force-overwrite true ./build/bevfusion $DEBUG_DATA $DEBUG_MODEL $DEBUG_PRECISION
