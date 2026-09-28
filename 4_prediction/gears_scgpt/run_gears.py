#!/usr/bin/env python3
"""Train GEARS on one CV fold and predict that fold's test perturbations.

Writes, for every test condition, the mean observed expression and the GEARS
prediction (both ABSOLUTE post-perturbation expression, not effects):

    <outdir>/<dataset>_<fold>_gears_post-gt.csv
    <outdir>/<dataset>_<fold>_gears_post-pred.csv

Usage
-----
    python run_gears.py --dataset VCC-cv5 --seed 3
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_data import get_pert_data  # first: puts 4_prediction/ on sys.path
import paths
from gears import GEARS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='<build>-cv5')
parser.add_argument('--seed', required=True, type=int, help='CV fold, 1-5')
parser.add_argument('--data_dir', default=None,
                    help='GEARS data root (default: paths.GEARS_DATA_DIR)')
parser.add_argument('--outdir', default=str(paths.GEARS_PRED_DIR))
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--hiddendim', default=64, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--load_model', action='store_true',
                    help='predict from a saved checkpoint instead of training')
args = parser.parse_args()

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset, seed=args.seed,
                              data_dir=args.data_dir)

    # Ref: https://github.com/snap-stanford/GEARS/blob/719328bd56745ab5f38c80dfca55cfd466ee356f/demo/model_tutorial.ipynb
    pert_data.get_dataloader(batch_size=args.batchsize,
                             test_batch_size=args.batchsize)
    model_path = f'{args.outdir}/checkpoints/gears_seed{args.seed}_{args.dataset}'
    gears_model = GEARS(pert_data, device=f'cuda:{args.device}',
                        weight_bias_track=False,
                        proj_name='pertnet',
                        exp_name='pertnet')
    gears_model.model_initialize(hidden_size=args.hiddendim)
    if args.load_model and os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        gears_model.load_pretrained(model_path)
    else:
        # without a test loader, GEARS.train() skips its own test-set evaluation;
        # the test predictions are written below
        gears_model.dataloader.pop('test_loader', None)
        gears_model.train(epochs=args.epochs, lr=args.lr)
        Path(f'{args.outdir}/checkpoints').mkdir(parents=True, exist_ok=True)
        gears_model.save_model(model_path)

    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test']
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train']

    unique_conds = list(set(test_adata.obs['condition'].astype(str).unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    processed_conds = []
    train_counts = []
    skipped_conds = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')
        if 'ctrl' in gene_list:
            gene_list.remove('ctrl')

        # GEARS cannot predict a gene outside its perturbation graph
        unsupported = [g for g in gene_list if g not in gears_model.pert_list]
        if unsupported:
            print(f'Skipping {condition}: {unsupported} not in GEARS.pert_list')
            skipped_conds.append(condition)
            continue

        adata_condition = test_adata[test_adata.obs['condition'] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[0]

        # number of this condition's genes seen as single perturbations in training
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                n_train += 1
        train_counts.append(n_train)

        gears_pred = list(gears_model.predict([gene_list]).values())[0]
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = gears_pred
        processed_conds.append(condition)

    if skipped_conds:
        print(f'Skipped {len(skipped_conds)}/{len(unique_conds)} conditions not in '
              f'GEARS.pert_list: {skipped_conds}')

    index = pd.MultiIndex.from_tuples(list(zip(processed_conds, train_counts)),
                                      names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_gears_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_gears_post-pred.csv')
