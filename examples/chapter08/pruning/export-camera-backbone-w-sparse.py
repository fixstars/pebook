import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify


from torch import nn
from apex.contrib.sparsity import ASP
import lean.quantize as quantize

def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument('--ckpt', type=str, default='qat/ckpt/bevfusion_ptq.pth')
    args = parser.parse_args()
    return args

class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)

        def get_cam_feats(self, x, d):
            B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)
            x = torch.cat([d, x], dim=1)
            x = self.depthnet(x)

            depth = x[:, : self.D].softmax(dim=1)
            feat  = x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1)
            return feat, depth
        
        return get_cam_feats(self.model.encoders.camera.vtransform, feat, depth)

# CUDA-BEVFusionのpytorch modelのQuantConv2dだとsparse化が効かないのでnn.Conv2dに変換
def convert_quant_to_fp32(module):
    for name, child in module.named_children():
        if child.__class__.__name__ == "QuantConv2d":
            print(f"Replacing {name} ({child.__class__.__name__}) with nn.Conv2d")

            weight = child.weight.cpu().dequantize() if hasattr(child.weight, "dequantize") else child.weight
            bias = child.bias.cpu() if child.bias is not None else None

            # 新しい Conv2d レイヤーを作成
            new_conv = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=bias is not None
            )
            new_conv.weight.data.copy_(weight)
            if bias is not None:
                new_conv.bias.data.copy_(bias)

            # モジュールを置換
            setattr(module, name, new_conv)

        else:
            convert_quant_to_fp32(child)

def main():
    args = parse_args()

    model  = torch.load(args.ckpt).module

    quantize.disable_quantization(model).apply()

    allowed_layer_names = []
    for name, mod in model.named_modules():
        if "encoders.camera.backbone" in name:
            allowed_layer_names.append(name)

    convert_quant_to_fp32(model)
    model = model.cuda()

    ASP.init_model_for_pruning(
        model,
        mask_calculator='m4n2_1d',
        whitelist=(torch.nn.Conv2d, torch.nn.Linear),
        allow_recompute_mask=False,
        allow_permutation=False,
        allowed_layer_names = allowed_layer_names,
    )

    ASP.compute_sparse_masks()
        
    data = torch.load("example-data/example-data.pth")
    img = data["img"].data[0].cuda()
    points = [i.cuda() for i in data["points"].data[0]]

    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval()
    depth = torch.zeros(len(points), img.shape[1], 1, img.shape[-2], img.shape[-1]).cuda()

    save_root = f"qat/onnx_fp16"
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone_fp16_w_sparse.onnx"
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        onnx.save(onnx_simp, camera_backbone_onnx)

if __name__ == "__main__":
    main()
