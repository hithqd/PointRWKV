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
| Pre-training | ShapeNet |                                                             | - | [model](https://drive.google.com/file/d/1QXB1msBljSOPJhx5sGYpueOdCrY0yaCO/view?usp=sharing) |
| Classification | ModelNet40 | 94.66% | [model](https://drive.google.com/file/d/1iMN-iAGjKWAUpAoIOqaS9e_CI_wk5nhE/view?usp=sharing) | 96.16% | [model](https://drive.google.com/file/d/11iBDSwdTIpHldUGWIsFp9orbCwNf69fB/view?usp=sharing) |
| Classification | ScanObjectNN | 92.88% | [model](https://drive.google.com/file/d/1DQx_5t9DNSIT11zLh1LZWJ5I3zgDXfmM/view?usp=sharing) | 93.05% | [model](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain.pth) |
| Part Segmentation | ShapeNetPart | - | - | 86.0% mIoU | [model](https://github.com/LMD0311/PointMamba/releases/download/ckpts/part_seg_pretrain.pth) |

## Qualitative Results
![image](../main/assets/vis.png)
