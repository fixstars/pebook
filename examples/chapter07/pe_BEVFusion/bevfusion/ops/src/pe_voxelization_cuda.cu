// Copyright (c) Fixstars. All rights reserved.
// modify from
// https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/projects/BEVFusion/bevfusion/ops/voxel/src/voxelization_cuda.cu
#include "./voxelization_cuda.cu"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace voxelization
{

    int hard_voxelize_gpu_pe(const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
                             at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
                             const std::vector<float> coors_range, const int max_points, const int max_voxels,
                             const int NDim = 3)
    {
        // current version tooks about 0.04s for one frame on cpu
        // check device
        CHECK_INPUT(points);

        at::cuda::CUDAGuard device_guard(points.device());

        const int num_points_raw = points.size(0);
        const int num_points = num_points_raw + 1;
        const int num_features = points.size(1);

        const float voxel_x = voxel_size[0];
        const float voxel_y = voxel_size[1];
        const float voxel_z = voxel_size[2];
        const float coors_x_min = coors_range[0];
        const float coors_y_min = coors_range[1];
        const float coors_z_min = coors_range[2];
        const float coors_x_max = coors_range[3];
        const float coors_y_max = coors_range[4];
        const float coors_z_max = coors_range[5];

        const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
        const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
        const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

        // map points to voxel coors
        at::Tensor temp_coors = at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

        dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points_raw, 512), 4096));
        dim3 block(512);

        // 1. link point to corresponding voxel coors
        AT_DISPATCH_ALL_TYPES(
            points.scalar_type(), "hard_voxelize_kernel", ([&] {
                dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    points.contiguous().data_ptr<scalar_t>(), temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
                    voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max, grid_x,
                    grid_y, grid_z, num_points_raw, num_features, NDim);
            }));
        // cudaDeviceSynchronize();
        AT_CUDA_CHECK(cudaGetLastError());
        // 番兵を追加
        temp_coors[num_points_raw] = -1;

        // 2. determined voxel's coor index
        //    各点について、その点が所属するvoxelのidを表す、coor_to_voxelidx を計算する

        // at::unique_dimにはSynchronizeがある
        // point_to_coorid を別の手段で計算することで取り除くことは可能。複雑なので省略。
        at::Tensor point_to_coorid;  // 元non-deterministicコードの coors_map
        at::Tensor dummy_0, dummy_2;
        const auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);
        std::tie(dummy_0, point_to_coorid, dummy_2) = at::unique_dim(coors_clean, 0, true, true, false);

        // unique_dimはsortを含むため、coorid==0 は無効値

        // unique_dimはsortを含むため、
        // point_to_coorid をそのままpoint_to_voxelidxとして使うと、座標的な偏りが生じる
        // pointsの順番での先着順でpoint_to_voxelidxを取るようにする

        // 同coorid内で、1番目となる点を探す
        const auto temp_arange = at::arange({num_points}, points.options().dtype(at::kInt));
        auto coorid_to_first_point_idx = at::index_reduce(at::full({num_points + 1}, num_points, temp_arange.options()),
                                                          0, point_to_coorid, temp_arange, "amin", true);
        // 無効値を取り除きつつ、
        // 同じcooridの中で1番目となる点のみがTrueとなるマスクを作る
        coorid_to_first_point_idx[0] = num_points;
        const auto points_to_first_point_idx = coorid_to_first_point_idx.index_select(0, point_to_coorid);
        const auto points_to_is_first_pts_in_same_coorid = points_to_first_point_idx == temp_arange;
        // voxelidxを配列の先頭から順に振る
        auto coor_to_voxelidx = points_to_is_first_pts_in_same_coorid.cumsum(0, at::kLong);
        coor_to_voxelidx -= 1;
        // この段階だとpoints_to_is_first_pts_in_same_coorid の要素しか正しい値になっていないので、
        // 正しい値をブロードキャストする
        // 無効値がout_of_rangeにならないようにする
        coor_to_voxelidx = at::cat({coor_to_voxelidx, at::full({1}, -1, coor_to_voxelidx.options())}, 0);
        coor_to_voxelidx = coor_to_voxelidx.index_select(0, points_to_first_point_idx);
        // max_voxels に制限する
        coor_to_voxelidx = coor_to_voxelidx.masked_fill(coor_to_voxelidx >= max_voxels, -1);
        coor_to_voxelidx = coor_to_voxelidx.to(at::kInt).contiguous();

        // 3. 各点について、その点が所属するvoxel内でのその点のidを表す、point_pos_in_voxelを求める
        // point_pos_in_voxelは、元コードでは point_to_voxelidx
        const auto sorted_to_point = at::argsort(coor_to_voxelidx, true, 0, false);  // stable sort
        const auto point_to_sorted = at::empty_like(temp_arange).scatter_(0, sorted_to_point, temp_arange);
        // 無効値がout_of_rangeにならないようにする
        const auto point_to_sorted_pad = at::cat({point_to_sorted, at::zeros({1}, coor_to_voxelidx.options())}, 0);
        auto point_pos_in_voxel = point_to_sorted - point_to_sorted_pad.index_select(0, points_to_first_point_idx);
        // 無効 voxels を除去
        point_pos_in_voxel = point_pos_in_voxel.masked_fill(coor_to_voxelidx < 0, -1);
        // max_points に制限する
        point_pos_in_voxel = point_pos_in_voxel.masked_fill(point_pos_in_voxel >= max_points, -1);
        point_pos_in_voxel = point_pos_in_voxel.to(at::kInt).contiguous();

        // 残りの値を計算
        const auto voxel_num_int = coor_to_voxelidx.max().to(at::kInt).to(at::kCPU).item<int>() + 1;
        const auto num_points_per_voxel_pad = at::index_reduce(
            point_pos_in_voxel.new_empty({num_points + 1}), 0,
            coor_to_voxelidx.masked_fill(coor_to_voxelidx < 0, num_points), point_pos_in_voxel + 1, "amax", false);
        num_points_per_voxel.slice(0, 0, voxel_num_int).copy_(num_points_per_voxel_pad.slice(0, 0, voxel_num_int));

        // 4. copy point features to voxels
        // Step 4 & 5 could be parallel
        auto pts_output_size = num_points * num_features;
        dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
        dim3 cp_block(512);
        AT_DISPATCH_ALL_TYPES(
            points.scalar_type(), "assign_point_to_voxel", ([&] {
                assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    pts_output_size, points.contiguous().data_ptr<float>(),
                    point_pos_in_voxel.contiguous().data_ptr<int>(), coor_to_voxelidx.contiguous().data_ptr<int>(),
                    voxels.contiguous().data_ptr<float>(), max_points, num_features, num_points, NDim);
            }));
        //   cudaDeviceSynchronize();
        AT_CUDA_CHECK(cudaGetLastError());

        // 5. copy coors of each voxels
        auto coors_output_size = num_points * NDim;
        dim3 coors_cp_grid(std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
        dim3 coors_cp_block(512);
        AT_DISPATCH_ALL_TYPES(
            points.scalar_type(), "assign_point_to_voxel", ([&] {
                assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    coors_output_size, temp_coors.contiguous().data_ptr<int>(),
                    point_pos_in_voxel.contiguous().data_ptr<int>(), coor_to_voxelidx.contiguous().data_ptr<int>(),
                    coors.contiguous().data_ptr<int>(), num_points, NDim);
            }));
        // cudaDeviceSynchronize();
        AT_CUDA_CHECK(cudaGetLastError());

        return voxel_num_int;
    }
}  // namespace voxelization
