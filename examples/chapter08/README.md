# 第08章のサンプルコード

## 実行環境
- デバイス: Jetson AGX Orin（Power Mode: 50W）
- TensorRT: 10.3.0.30-1+cuda12.5
- NVIDIA Nsight Systems: 2024.5.4.34-245434855735
- NVIDIA Nsight Compute: 2024.3.1.0
- Apex: 91fcaaf718306e3441e9a8369aa4553eea2e326b (commit hash)
- Lidar AI Solution（CUDA-BEVFusion）: 30dac933e540c3c08052dc62c51f55e803df8dac (commit hash)

### CUDA-BEVFusionの準備
ベースラインとしたCUDA-BEVFusionがあるリポジトリをcloneします。
以下のコマンドを実行対象デバイス、および枝刈り処理を行うホストPCの双方で実行してください。

```bash
git clone --recursive https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git
cd Lidar_AI_Solution
git checkout 30dac933e540c3c08052dc62c51f55e803df8dac
```

以降、`Lidar_AI_Solution/CUDA-BEVFusion/`のパスを`${CUDA-BEVFusion_ROOT}`と表記します。

## 各種サンプルコードについて
このディレクトリ以下に以下の内容が含まれています。

### modify_CUDA-BEVFusion_src
CUDA-BEVFusionのソースコードを本書記載の内容に修正したファイル群です。
`${CUDA-BEVFusion_ROOT}/src/`にこれらのファイルを上書きコピーして使用してください。
これらのファイルには以下の変更が含まれています。
- サブグラフごとにNVTXアノテーションの付与
- 画像データのデバイスへの転送処理とLiDAR Backbone処理の並列化

### quantize
このディレクトリには、CameraBackboneに量子化を適用してプロファイルを取得するスクリプトがあります。

実行対象のデバイスにて、[CUDA-BEVFusionのREADME](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md)の`Quick Start for Inference`を完了させた後、`build_quantized_camera_backbone.sh`を`${CUDA-BEVFusion_ROOT}`にコピーして以下のコマンドを実行してください。

```bash
cd ${CUDA-BEVFusion_ROOT}
bash ./build_quantized_camera_backbone.sh
```

実行すると`prof_quantized.nsys-rep`というファイルが生成されます。このファイルをNVIDIA Nsight Systemsで開くとプロファイリング結果を確認できます。


### pruning
このディレクトリには、CameraBackboneに枝刈りを適用してプロファイルを取得するスクリプトがあります。

`export-camera-backbone-w-sparse.py`は枝刈り適用済みのONNXファイルを生成するスクリプトです。
まず、ホストPCなどで[CUDA-BEVFusionのqat/README](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/qat/README.md)に従い、`Configuring the bevfusion runtime environment`と`Download model.zip and nuScenes-example-data.zip`まで済ませてください。

その上で、`${CUDA-BEVFusion_ROOT}/qat/`に`export-camera-backbone-w-sparse.py`をコピーし、以下のdocker container内でコマンドを実行してください。

```bash
# Apexのインストール
cd ${CUDA-BEVFusion_ROOT}
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 91fcaaf718306e3441e9a8369aa4553eea2e326b
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
pip install numba

# pytorch-quantizationでエラーになる場合は、以下の手順で動作確認済みバージョンをインストール
pip uninstall pytorch-quantization
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/
git checkout release/8.6
cd tools/pytorch-quantization
python setup.py install

# 枝刈りの実行
cd ${CUDA-BEVFusion_ROOT}
python3 qat/export-camera-backbone-w-sparse.py --ckpt=model/resnet50int8/bevfusion_ptq.pth
```

実行後、`${CUDA-BEVFusion_ROOT}/qat/onnx_fp16/`以下に`camera.backbone_fp16_w_sparse.onnx`というファイルが生成されます。
実行対象のデバイスにて[CUDA-BEVFusionのREADME](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md)の`Quick Start for Inference`を完了させた後に、このファイルを実行対象デバイスの`${CUDA-BEVFusion_ROOT}/model/resnet50/`以下にコピーしてください。

そして、`build_pruned_camera_backbone.sh`を実行対象デバイスの`${CUDA-BEVFusion_ROOT}`にコピーして以下のコマンドを実行してください。

```bash
cd ${CUDA-BEVFusion_ROOT}
bash ./build_pruned_camera_backbone.sh
```

実行すると`prof_pruned.nsys-rep`というファイルが生成されます。このファイルをNVIDIA Nsight Systemsで開くとプロファイリング結果を確認できます。
