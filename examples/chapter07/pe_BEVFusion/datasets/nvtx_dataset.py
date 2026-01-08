# Copyright (c) Fixstars. All rights reserved.
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import CBGSDataset
import nvtx

@DATASETS.register_module()
class NVTXCBGSDataset(CBGSDataset):
    def __getitem__(self, idx: int) -> dict:
        with nvtx.annotate("__getitem__"):
            return super().__getitem__(idx)


