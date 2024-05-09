

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['stock', 'weather', 'stock_pretrain', 'stock_finetune','stock_predict'
        ]

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False



  
    if params.dset == 'stock_pretrain':
        root_path = '/home/userroot/dev/zms/etl/index_300'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=ConcatStockDataset,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': '*.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'stock_finetune':
        root_path = '/home/userroot/dev/zms/etl/index_300_finetune'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=ConcatStockDataset,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': '*.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'stock_predict':
        root_path = '/home/userroot/dev/zms/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Predict_Stock,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'etl/index_300_test/sz300750.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
