# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.datestr = None
        self.env_config = env_config

        train = pd.read_csv('./scripts/train.csv')
        test = pd.read_csv('./scripts/test.csv')

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_file = open("./scripts/list.txt", 'r')
        feature_map = get_feature_map(feature_file)
        fc_struc = get_fc_graph_struc(feature_file)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)

    def edge_indices_from_A(self, A):
        row_indices, col_indices = torch.where(A > 0)
        edge_indices = torch.stack((row_indices, col_indices), dim=0)
        return edge_indices.to(torch.long)

    def run(self):
        model_save_path = self.get_save_path()[0]

        self.train_log = train(self.model, model_save_path,
                               config=train_config,
                               train_dataloader=self.train_dataloader,
                               val_dataloader=self.val_dataloader,
                               feature_map=None,
                               test_dataloader=self.test_dataloader,
                               test_dataset=self.test_dataset,
                               train_dataset=self.train_dataset,
                               dataset_name="SWaT"
                               )
        # test
        self.model.load_state_dict(torch.load(model_save_path, weights_only=True))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        paths = [
            f'./pretrained/{dir_path}/best_model.pt',
            f'./results/{dir_path}/result.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                      shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                    shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')


if __name__ == "__main__":
    random_seed = 6
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)


    train_config = {
        'batch': 32,
        'epoch': 30,
        'slide_win': 5,
        'dim': 64,
        'slide_stride': 1,
        'seed': random_seed,
        'out_layer_num': 3,
        'out_layer_inter_dim': 64,
        'val_ratio': 0.1,
        'topk': 15,
        'lr': 0.001
    }

    env_config = {
        'save_path': "save_models",
        'report': 'best',
        'device': 'cuda',
        'load_model_path': ''
    }

    main = Main(train_config, env_config, debug=False)
    main.run()





