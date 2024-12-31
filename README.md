# [AAAI2025]PointRWKV:Efficient RWKV-Like Model for Hierarchical Point Cloud Learning

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2405.15214" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://hithqd.github.io/projects/PointRWKV/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
</div>

## Abstract
Transformers have revolutionized the point cloud learning task, but the quadratic complexity hinders its extension to long sequences. This puts a burden on limited computational resources. The recent advent of RWKV, a fresh breed of deep sequence models, has shown immense potential for sequence modeling in NLP tasks. In this work, we present PointRWKV, a new model of linear complexity derived from the RWKV model in the NLP field with the necessary adaptation for 3D point cloud learning tasks. Specifically, taking the embedded point patches as input, we first propose to explore the global processing capabilities within PointRWKV blocks using modified multi-headed matrix-valued states and a dynamic attention recurrence mechanism. To extract local geometric features simultaneously, we design a parallel branch to encode the point cloud efficiently in a fixed radius near-neighbors graph with a graph stabilizer. Furthermore, we design PointRWKV as a multi-scale framework for hierarchical feature learning of 3D point clouds, facilitating various downstream tasks. Extensive experiments on different point cloud learning tasks show our proposed PointRWKV outperforms the transformer- and mamba-based counterparts, while significantly saving about 42\% FLOPs, demonstrating the potential option for constructing foundational 3D models. 
## Method
![image](../main/assets/architecture.png)  

## Requirements

```
pip install -r requirements.txt
```
```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
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

## Dataset

The overall directory structure should be:
```
│PointRWKV/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ModelNet Few-shot Dataset:
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55/34 Dataset:

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md).

### ShapeNetPart Dataset:

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 

## Training
We provide `./scripts/dist_train.sh` that is able to train a model with the specified config file. Run it with:
```
sh ./scripts/dist_train.sh 1 --cfg_file ./PATH/TO/YAML --workers 4
```
For more options, see ./scripts/dist_train.sh.

## Inference with Pre-trained Models
Refer to `./scripts/dist_test.sh` that is able to test with a pretrained model. Run it with:
```
sh ./scripts/dist_test.sh 1 --cfg_file ./PATH/TO/YAML --workers 4 --ckpt /PATH/TO/checkpoint_file --eval_tag test_out --batch_size 8 --save_to_file [--other-options]
```

## Qualitative Results
![image](../main/assets/vis.png)

## Acknowledgement

This project is based on Point-BERT ([paper](https://arxiv.org/abs/2111.14819), [code](https://github.com/lulutang0608/Point-BERT?tab=readme-ov-file)), Point-MAE ([paper](https://arxiv.org/abs/2203.06604), [code](https://github.com/Pang-Yatian/Point-MAE)), PointM2AE ([paper](https://arxiv.org/abs/2205.14401), [code](https://github.com/ZrrSkywalker/Point-M2AE)). Thanks for their wonderful works.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@article{he2024pointrwkv,
  title={PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning},
  author={He, Qingdong and Zhang, Jiangning and Peng, Jinlong and He, Haoyang and Li, Xiangtai and Wang, Yabiao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2405.15214},
  year={2024}
}
