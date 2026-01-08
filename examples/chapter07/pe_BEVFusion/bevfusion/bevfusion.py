# Copyright (c) Fixstars. All rights reserved.
import torch
from torch import Tensor
from torch.nn import functional as F

# from deepspeed.profiling.flops_profiler import FlopsProfiler
from mmdet3d.registry import MODELS
from mmdetection3d.projects.BEVFusion.bevfusion import BEVFusion

@MODELS.register_module()
class PE_BEVFusion_model(BEVFusion):
    pass
    # DEBUG
    # @torch.no_grad()
    # def voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret_cpu = self.pts_voxel_layer(res.cpu())
    #         ret = self.pts_voxel_layer(res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #             f_cpu, c_cpu, n_cpu = ret_cpu
    #             assert torch.allclose(c.cpu(), c_cpu), f"torch.allclose(c, c_cpu), {c}/{c_cpu}"
    #             assert torch.allclose(n.cpu(), n_cpu), f"torch.allclose(n, n_cpu), {n}/{n_cpu}"
    #             assert torch.allclose(f.cpu(), f_cpu), f"torch.allclose(f.cpu(), f_cpu), {n}/{n_cpu}"
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode='constant', value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(
    #                 dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

