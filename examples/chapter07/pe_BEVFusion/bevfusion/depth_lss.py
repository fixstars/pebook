# Copyright (c) Fixstars. All rights reserved.
# modify from https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/projects/BEVFusion/bevfusion/depth_lss.py
# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
import torch.linalg
import torch.amp
import nvtx

from mmdet3d.registry import MODELS
from mmdetection3d.projects.BEVFusion.bevfusion.depth_lss import DepthLSSTransform
from mmdetection3d.projects.BEVFusion.bevfusion.ops import bev_pool

@MODELS.register_module()
class PE_DepthLSSTransform(DepthLSSTransform):

    @nvtx.annotate("get_geometry", color="blue")
    @torch.autocast("cuda", enabled=False)
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        # PATCH BEGIN
        # 精度が下がらないようにdoubleにしておく
        # assert camera2lidar_rots.dtype == torch.float
        # assert camera2lidar_trans.dtype == torch.float
        # assert intrins.dtype == torch.float
        # assert post_rots.dtype == torch.float
        # assert post_trans.dtype == torch.float
        # camera2lidar_rots = camera2lidar_rots.to(torch.double)
        # camera2lidar_trans = camera2lidar_trans.to(torch.double)
        # intrins = intrins.to(torch.double)
        # post_rots = post_rots.to(torch.double)
        # post_trans = post_trans.to(torch.double)
        # if 'extra_rots' in kwargs:
        #     assert kwargs['extra_rots'].dtype == torch.float
        #     kwargs['extra_rots'] = kwargs['extra_rots'].to(torch.double)
        # if 'extra_trans' in kwargs:
        #     assert kwargs['extra_trans'].dtype == torch.float
        #     kwargs['extra_trans'] = kwargs['extra_trans'].to(torch.double)
        # PATCH END

        # PATCH BEGIN: DEBUG
        # with torch.no_grad():
        #     expect = super().get_geometry(
        #         camera2lidar_rots,
        #         camera2lidar_trans,
        #         intrins,
        #         post_rots,
        #         post_trans,
        #         **kwargs,
        #     )
        #     expect = expect.to(torch.float)
        # PATCH END


        B, N, _ = camera2lidar_trans.shape
        D, H, W, _ = self.frustum.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)

        # PATCH BEGIN: use torch.linalg.solve instead of torch.inverse
        # 1. To avoid synchronize CUDA stream when using torch.inverse
        # 2. To avoid slow torch.bmm for (*,3,3)x(*,3,1) matrix
        post_rots_inv, solve_info0 = torch.linalg.inv_ex(post_rots)
        post_rots_inv = post_rots_inv.view(B, N, 3, 3)
        points = torch.einsum("bnxy,bndhwy->bndhwx", post_rots_inv, points).unsqueeze(-1)
        # PATCH END

        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        # PATCH BEGIN: use torch.linalg.solve_ex instead of torch.inverse
        combine, solve_info1 = torch.linalg.solve_ex(intrins, camera2lidar_rots, left=False)
        # PATCH END
        # PATCH BEGIN: use torch.einsum instead of torch.matmul
        # (B x N x 1 x 1 x 1 x 3 x 3) * (B x N x D x H x W x 3 x 1) -> (B x N x D x H x W x 3 x 1).squeeze(-1)
        points = points.squeeze(-1).contiguous()
        points = torch.einsum("bnxy,bndhwy->bndhwx", combine.view(B, N, 3, 3), points)
        # PATCH END
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            # PATCH BEGIN: use torch.einsum instead of torch.matmul
            # (B x 1 x 1 x 1 x 1 x 3 x 3) * (B x N x D x H x W x 3 x 1) -> (B x N x D x H x W x 3 x 1).squeeze(-1)
            points = torch.einsum("bxy,bndhwy->bndhwx", extra_rots.view(B, 3, 3), points).contiguous()
            # PATCH END
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        # PATCH BEGIN
        # 精度を元に戻す
        points = points.to(torch.float)
        # PATCH END

        # PATCH BEGIN: DEBUG
        # assert torch.all(solve_info0 == 0)
        # assert torch.all(solve_info1 == 0)
        # atol=1e-4 # float
        # rtol=1e-2 # float
        # # atol=1e-8 # double
        # # rtol=1e-5 # double
        # a_err = (expect - points).abs()
        # r_err = a_err / expect.abs()
        # total_err = (a_err >atol) & (r_err>rtol)
        # if total_err.sum().item()>0:
        #     print(a_err[total_err], r_err[total_err])
        # assert torch.allclose(expect, points, atol=atol, rtol=rtol)
        # PATCH END
        return points

    @nvtx.annotate("bev_pool", color="blue")
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        with nvtx.annotate("kept", color="blue"):
            kept = ((geom_feats[:, 0] >= 0)
                    & (geom_feats[:, 0] < self.nx[0])
                    & (geom_feats[:, 1] >= 0)
                    & (geom_feats[:, 1] < self.nx[1])
                    & (geom_feats[:, 2] >= 0)
                    & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        with nvtx.annotate("bev_pool", color="red"):
            x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @nvtx.annotate("_BaseDepthTransform_forward", color="blue")
    def _BaseDepthTransform_forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        # PATCH BEGIN: to(devide)ではなく、torch.zerosにdevice引数を使う
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size, device=points[0].device)
        # PATCH END

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            # PATCH BEGIN: torch.linalg.invをtorch.linalg.inv_exに変更
            cur_coords, _ = torch.linalg.inv_ex(cur_lidar_aug_matrix[:3, :3])
            # PATCH END
            cur_coords = cur_coords.matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            # PATCH BEGIN: torch.stackを使うように変更
            cur_coords = torch.stack((cur_coords[..., 1], cur_coords[..., 0]), dim=-1)
            # PATCH END

            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))

            depth = depth.to(dist.dtype)
            out_img = ~on_img
            # PATCH BEGIN: nonzeroを使わずに処理
            masked_coords = cur_coords.long()
            # 2次元座標を1次元にならす
            masked_coords[..., 0] *= self.image_size[1]
            masked_coords = masked_coords.sum(-1)
            masked_coords.masked_fill_(out_img, self.image_size[0] * self.image_size[1]) # 無効indexは末尾(padding)を指すようにする
            # for c in range(on_img.shape[0]):
            #   depth[b, c, 0, masked_coords[c]] = dist[c]
            # padding付きの出力先を作成
            depth_padding = depth.new_empty(img.shape[1], self.image_size[0] * self.image_size[1] + 1) # +1 for padding
            depth_padding[:, :-1].view(img.shape[1], *self.image_size).copy_(depth[b, :, 0]) # fill deafult value
            # 各要素を指定されたindexに代入する。無効値は末尾のpaddingに代入される
            depth_padding.scatter_(1, masked_coords, dist)
            depth[b, :, 0].copy_(depth_padding[:, :-1].view(img.shape[1], *self.image_size))
            # PATCH END

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x

    def forward(self, *args, **kwargs):
        x = self._BaseDepthTransform_forward(*args, **kwargs)
        x = self.downsample(x)
        return x
