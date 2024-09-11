# PointRWKV:Efficient RWKV-Like Model for Hierarchical Point Cloud Learning

## Method
![image](../main/assets/architecture.png)  

## Requirements

```
pip install -r requirements.txt
```
```

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

```
## Model Zoo
| Task | Dataset | Acc.(Scratch) | Download (Scratch) | Acc.(pretrain) | Download (Finetune) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Pre-training | ShapeNet |                                                             | - | [model](https://drive.google.com/file/d/1QXB1msBljSOPJhx5sGYpueOdCrY0yaCO/view?usp=sharing) |
| Classification | ModelNet40 | 94.66% | [model](https://drive.google.com/file/d/1iMN-iAGjKWAUpAoIOqaS9e_CI_wk5nhE/view?usp=sharing) | 96.16% | [model](https://drive.google.com/file/d/11iBDSwdTIpHldUGWIsFp9orbCwNf69fB/view?usp=sharing) |
| Classification | ScanObjectNN | 92.88% | [model](https://drive.google.com/file/d/1DQx_5t9DNSIT11zLh1LZWJ5I3zgDXfmM/view?usp=sharing) | 93.05% | [model](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain.pth) |
| Part Segmentation | ShapeNetPart | - | - | 90.26% mIoU | [model](https://drive.google.com/file/d/1hQnB8uGzFGXUWXzM9ihjobIE-O9h9c2v/view?usp=sharing) |

<div  align="center">    
 <img src="../main/assets/flops.png" width = "488"  align=center />
</div>

## Qualitative Results
![image](../main/assets/vis.png)

## Acknowledgement

This project is based on Point-BERT ([paper](https://arxiv.org/abs/2111.14819), [code](https://github.com/lulutang0608/Point-BERT?tab=readme-ov-file)), Point-MAE ([paper](https://arxiv.org/abs/2203.06604), [code](https://github.com/Pang-Yatian/Point-MAE)), PointM2AE ([paper](https://arxiv.org/abs/2205.14401), [code](https://github.com/ZrrSkywalker/Point-M2AE)). Thanks for their wonderful works.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@article{he2024pointrwkv,
  title={PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning},
  author={He, Qingdong and Zhang, Jiangning and Peng, Jinlong and He, Haoyang and Wang, Yabiao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2405.15214},
  year={2024}
}
