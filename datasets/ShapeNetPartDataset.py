"""ShapeNetPart dataset for part segmentation."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import print_log


@DATASETS.register_module()
class ShapeNetPart(Dataset):
    """ShapeNetPart dataset for 3D part segmentation.
    
    Each shape has a category label and per-point part labels.
    16 shape categories, 50 part classes total.
    """
    
    # Category name to synset ID mapping
    cat_to_synset = {
        'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340',
        'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776',
        'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649',
        'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390',
        'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987',
        'Table': '04379243',
    }
    
    # Number of parts per category
    seg_classes = {
        'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7],
        'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
        'Knife': [22, 23], 'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29], 'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37], 'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }
    
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.get('N_POINTS', 2048)
        self.subset = config.subset
        self.num_classes = 16
        self.num_parts = 50
        
        # Build synset to category mapping
        self.synset_to_cat = {v: k for k, v in self.cat_to_synset.items()}
        
        # Read category file
        synsetoffset_path = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(synsetoffset_path, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        
        # Category to index
        self.classes_original = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        # Read split
        split_file = os.path.join(self.root, 'train_test_split',
                                   f'shuffled_{self.subset}_file_list.json')
        with open(split_file, 'r') as f:
            split_ids = set([x.strip().split('/')[-1] for x in json.load(f)])
        
        # Collect data paths
        self.datapath = []
        for cat_name in sorted(self.cat.keys()):
            cat_dir = os.path.join(self.root, self.cat[cat_name])
            if not os.path.isdir(cat_dir):
                continue
            fns = sorted(os.listdir(cat_dir))
            for fn in fns:
                token = fn[:-4]  # remove .txt
                if token in split_ids:
                    self.datapath.append((cat_name, os.path.join(cat_dir, fn)))
        
        print_log(f'[ShapeNetPart] {self.subset}: {len(self.datapath)} shapes loaded',
                   logger='ShapeNetPart')
        
        # Build category to segment class mapping
        self.seg_start_index = {}
        for cat in self.seg_classes:
            self.seg_start_index[cat] = self.seg_classes[cat][0]
    
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        cat_name, fn = self.datapath[index]
        cls_idx = self.classes_original[cat_name]
        
        # Load point cloud (x, y, z, nx, ny, nz, label)
        data = np.loadtxt(fn).astype(np.float32)
        point_set = data[:, :3]
        normal = data[:, 3:6]
        seg = data[:, -1].astype(np.int64)
        
        # Normalize
        point_set = point_set - np.mean(point_set, axis=0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
        point_set = point_set / dist
        
        # Subsample
        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        
        point_set = point_set[choice]
        seg = seg[choice]
        
        # One-hot category label
        cls_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        cls_one_hot[cls_idx] = 1.0
        
        point_set = torch.from_numpy(point_set).float()
        seg = torch.from_numpy(seg).long()
        cls_one_hot = torch.from_numpy(cls_one_hot).float()
        
        return 'ShapeNetPart', 'sample', (point_set, cls_one_hot, seg, cls_idx)
