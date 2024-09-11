# PointRWKV:Efficient RWKV-Like Model for Hierarchical Point Cloud Learning

## Method
![image](../main/assets/architecture.png)  

## Requirements
Tested on:
PyTorch == 1.13.1;
python == 3.8;
CUDA == 11.7
```
pip install -r requirements.txt
```
```

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

```

## Quantitative Results
![image](../main/assets/flops.png)

| Task | Dataset | Acc.(Scratch) | Download (Scratch) | Acc.(pretrain) | Download (Finetune) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Pre-training | ShapeNet |                                                             | N.A. | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/pretrain.pth) |
| Classification | ModelNet40 | 92.4% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/modelnet_scratch.pth) | 93.6% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/modelnet_pretrain.pth) |
| Classification | ScanObjectNN | 88.30% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_scratch.pth) | 90.71% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain.pth) |
| Classification* | ScanObjectNN | \ | \ | 93.29% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain2.pth) |
| Classification | ScanObjectNN | 87.78% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objonly_scratch.pth) | 88.47% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objonly_pretrain.pth) |
| Classification* | ScanObjectNN | \ | \ | 91.91% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objonly_pretrain2.pth) |
| Classification | ScanObjectNN | 82.48% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_hardest_scratch.pth) | 84.87% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_hardest_pretrain.pth) |
| Classification* | ScanObjectNN | \ | \ | 88.17% | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_hardest_pretrain2.pth) |
| Part Segmentation | ShapeNetPart | 85.8% mIoU | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/part_seg_scratch.pth) | 86.0% mIoU | [here](https://github.com/LMD0311/PointMamba/releases/download/ckpts/part_seg_pretrain.pth) |

## Qualitative Results
![image](../main/assets/vis.png)
