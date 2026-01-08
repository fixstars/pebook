# Copyright (c) Fixstars. All rights reserved.
_base_ = [
    "../../mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
]
custom_imports = dict(
    imports=[
        "mmdetection3d.projects.BEVFusion.bevfusion",
        "pe_BEVFusion",
    ],
    allow_failed_imports=False,
)


custom_hooks = [
    dict(type="BenchmarkHook"),
    # dict(type="TorchProfilerHook"),
    # dict(type="NVTXHook"),
]
model = dict(
    type="pe_BEVFusion.PE_BEVFusion_model",
    data_preprocessor=dict(voxelize_cfg=dict(deterministic=False)),
    bbox_head=dict(type="pe_BEVFusion.PE_TransFusionHead"),
    view_transform=dict(type="pe_BEVFusion.PE_DepthLSSTransform"),
)
# この batch_size はマイクロバッチサイズ
train_dataloader = dict(batch_size = 8, dataset=dict(type="NVTXCBGSDataset"))
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(base_batch_size=64)
