# PointRWKV:Efficient RWKV-Like Model for Hierarchical Point Cloud Learning

## Method
![image](../main/assets/architecture.png)  

## 🎒 1. Requirements
Tested on:
PyTorch == 1.13.1;
python == 3.8;
CUDA == 11.7

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

# Mamba install
pip install causal-conv1d==1.1.1
```

## Quantitative Results
![image](../main/assets/flops.png)

## Qualitative Results
![image](../main/assets/vis.png)
