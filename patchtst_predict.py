import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_predict', type=str, default='stock_predict', help='dataset name')
parser.add_argument('--context_points', type=int, default=96, help='sequence length')
parser.add_argument('--target_points', type=int, default=32, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=5, help='patch length')
parser.add_argument('--stride', type=int, default=5, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dset_predict + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


# get available GPU devide
set_device()


def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def pred_func(weight_path):
    # get dataloader
    #dls = get_dls(args)
    root_path = '/home/userroot/dev/zms/'
    size = [args.context_points, 0, args.target_points]
    dataset = Predict_Stock(

                root_path= root_path,
                data_path= 'etl/index_300_test/sz300750.csv',
                features = args.features,
                scale = True,
                size =  size,
                use_time_features = False
                )
    dl = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=None,
        )
    nvars = dl.dataset[0][0].shape[1]
    model = get_model(nvars, args, head_type='prediction', weight_path=weight_path).to('cuda')
    # get callbacks
    cbs = [RevInCB(nvars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dl, model,cbs=cbs)
    #out  = learn.test(dl,  scores=[mse,mae])         # out: a list of [pred, targ, score]
    out  = learn.test(dl)
    print('predicts counts is ', len(learn.preds))
    #print('all predicts:', learn.preds)
    # save learn.preds to csv file
    args.predict_output_file = './predict_output.pt'
    pred_cnt = learn.preds.shape[0]
    pred_len = learn.preds.shape[1]
    pred_dim = learn.preds.shape[2]
    preds = learn.preds.reshape(-1, pred_dim)
    preds_inversed = dl.dataset.scaler.inverse_transform(preds)
    preds_inversed = preds_inversed.reshape(pred_cnt, pred_len, pred_dim)
    torch.save(preds_inversed, args.predict_output_file)
    print('predicts saved to ', args.predict_output_file)

    # print('score:', out[2])
    # # save results
    # pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':
    #args.pretrained_model = '/home/userroot/dev/zms/saved_models/stock/masked_patchtst/based_model/patchtst_pretrained_cw96_patch3_stride3_epochs-pretrain100_mask0.4_model1.pth'
        
    
    args.dset = args.dset_predict

    weight_path = '/home/userroot/dev/zms/saved_models/stock_finetune/masked_patchtst/based_model/stock_finetune_patchtst_linear-probe_cw96_tw32_patch5_stride5_epochs-finetune20_model1.pth'
    # Test
    out = pred_func(weight_path)        
    print('----------- Complete! -----------')
