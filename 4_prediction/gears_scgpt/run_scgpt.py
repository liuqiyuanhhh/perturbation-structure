#!/usr/bin/env python3
"""Fine-tune scGPT on one CV fold and predict that fold's test perturbations.

Starts from the pretrained whole-human checkpoint (``--modeldir``), fine-tunes on
the fold's training perturbations with early stopping on validation MSE, and
predicts each test perturbation from a pool of control cells.  Writes ABSOLUTE
post-perturbation expression, not effects:

    <outdir>/<dataset>_<fold>_scgpt_ft_post-gt.csv
    <outdir>/<dataset>_<fold>_scgpt_ft_post-pred.csv

Usage
-----
    python run_scgpt.py --dataset VCC-cv5 --seed 3 --finetune --train --epochs 20
    python run_scgpt.py --dataset VCC-cv5 --seed 3 --finetune \\
        --load_checkpoint <outdir>/checkpoints/scgpt_ft_seed3_VCC-cv5_epoch12
"""

import copy
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from torch_geometric.loader import DataLoader

from prepare_data import get_pert_data  # first: puts 4_prediction/ on sys.path
import paths
from gears.utils import create_cell_graph_dataset_for_prediction

from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="<build>-cv5")
parser.add_argument("--seed", required=True, type=int, help="CV fold, 1-5")
parser.add_argument("--data_dir", default=None,
                    help="GEARS data root (default: paths.GEARS_DATA_DIR)")
parser.add_argument("--outdir", default=str(paths.GEARS_PRED_DIR))
parser.add_argument("--device", default=0, type=int)
parser.add_argument("--finetune", action="store_true",
                    help="start from the pretrained checkpoint in --modeldir")
parser.add_argument("--train", action="store_true")

# model specific
parser.add_argument("--modeldir", default=str(paths.SCGPT_MODEL_DIR))
parser.add_argument("--modelfile", default=None,
                    help="pretrained weights (default: <modeldir>/best_model.pt)")
# Predict from an already fine-tuned checkpoint written by the torch.save below
# (best-validation weights), skipping training.  Use WITHOUT --train.
parser.add_argument("--load_checkpoint", default=None)

# default values are from the original scGPT implementation
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--lr", default=1e-4, type=float)

args = parser.parse_args()

model_params = {
    "load_model": args.modeldir,
    "model_file": args.modelfile,
    "load_param_prefixs": [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ],
    "embsize": 512,
    "d_hid": 512,
    "nlayers": 12,
    "nhead": 8,
    "n_layers_cls": 3,
    "dropout": 0.2,
    "use_fast_transformer": False,
}

trainer_params = {
    "mlm": True,  # masked language modeling
    "cls": False,  # celltype classification
    "cce": False,  # contrastive cell embedding
    "mvc": False,  # masked value prediction
    "ecs": False,  # elastic cell similarity
    "cell_emb_style": "cls",
    "mvc_decoder_style": "inner product, detach",
    "amp": True,
    "lr": args.lr,
    "batch_size": args.batchsize,
    "eval_batch_size": args.batchsize,
    "epochs": args.epochs,
    "schedule_interval": 1,
    "early_stop": 5,
    "log_interval": 100,
}

data_params = {
    "pad_tokens": "<pad>",
    "special_tokens": ["<pad>", "<cls>", "<eoc>"],
    "pad_value": 0,
    "pert_pad_id": 2,
    "n_hvg": 0,  # number of highly variable genes
    "include_zero_gene": "all",  # "all", "batch-wise", "row-wise", or False
    "max_seq_len": 1536,
    # number of control cells each test perturbation is predicted from
    # (None: all control cells)
    "control_pool_size": 100,
}


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """Load pretrained weights whose names start with one of ``prefix``."""
    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                print(f"[load_pretrained] Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    print(f"[load_pretrained] Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model


def scgpt_forward(
    batch_data,
    model,
    criterion,
    gene_ids,
    data_params,
    trainer_params,
    n_genes,
    device,
    test=False,
):
    """One forward pass; returns the loss (training) or the predictions (test)."""
    batch_data.to(device)
    x: torch.Tensor = batch_data.x

    batch_size = x.shape[0] // n_genes
    if x.dim() == 1:
        ori_gene_values = x.view(batch_size, n_genes)
    else:
        ori_gene_values = x[:, 0].view(batch_size, n_genes)

    # Perturbation flags, as in GEARS:
    # https://github.com/snap-stanford/GEARS/blob/719328bd56745ab5f38c80dfca55cfd466ee356f/gears/model.py#L151
    pert_flags = torch.zeros_like(ori_gene_values, dtype=torch.long, device=device)

    if batch_data.pert is not None:
        for i, p in enumerate(batch_data.pert):
            if isinstance(p, list):
                gene_list = p.copy()
                if "ctrl" in gene_list:
                    gene_list.remove("ctrl")
            else:
                gene_list = list(set(p.split("+")) - set(["ctrl"]))

            for g in gene_list:
                if g in data_params["genes"]:
                    pert_flags[i, data_params["genes"][g]] = 1

    if data_params["include_zero_gene"] in ["all", "batch-wise"]:
        if data_params["include_zero_gene"] == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]

        # during training, a random max_seq_len-gene subset per batch
        if not test and len(input_gene_ids) > data_params["max_seq_len"]:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[: data_params["max_seq_len"]]

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        if not test:
            target_gene_values = batch_data.y
            target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=input_values.device
        )

    with torch.cuda.amp.autocast(enabled=trainer_params["amp"]):
        output_dict = model(
            mapped_input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=trainer_params["cls"],
            CCE=trainer_params["cce"],
            MVC=trainer_params["mvc"],
            ECS=trainer_params["ecs"],
            do_sample=True,
        )
        output_values = output_dict["mlm_output"]

    if not test:
        masked_positions = torch.ones_like(input_values, dtype=torch.bool, device=input_values.device)
        loss = criterion(output_values, target_values, masked_positions)
        return loss

    return output_values


if __name__ == "__main__":
    set_seed(args.seed)
    device = f"cuda:{args.device}"

    model_name = "scgpt"
    if args.finetune:
        model_name += "_ft"
    else:
        model_params["load_model"] = None

    # Load data
    pert_data = get_pert_data(dataset=args.dataset, seed=args.seed, data_dir=args.data_dir)
    pert_data.get_dataloader(batch_size=args.batchsize, test_batch_size=args.batchsize)
    print(f"Data loaded for dataset {args.dataset} with seed {args.seed}. "
          f"Train samples: {len(pert_data.dataloader['train_loader'].dataset)}, "
          f"Val samples: {len(pert_data.dataloader['val_loader'].dataset)}, "
          f"Test samples: {len(pert_data.dataloader['test_loader'].dataset)}.")

    # Load metadata of the model and data
    if args.finetune and model_params["load_model"] is not None:
        print(f"Loading model and data metadata for finetuning from {model_params['load_model']}")
        model_dir = Path(model_params["load_model"])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        # Gene vocabulary; report how many genes in the data it covers
        vocab = GeneVocab.from_file(vocab_file)
        for s in data_params["special_tokens"]:
            if s not in vocab:
                vocab.append_token(s)

        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

        genes = pert_data.adata.var["gene_name"].tolist()

        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(f"Resume model from {model_file}, the model args will be overriden by "
              f"the config {model_config_file}.")
        model_params["embsize"] = model_configs["embsize"]
        model_params["nhead"] = model_configs["nheads"]
        model_params["d_hid"] = model_configs["d_hid"]
        model_params["nlayers"] = model_configs["nlayers"]
        model_params["n_layers_cls"] = model_configs["n_layers_cls"]
    else:
        model_file = None

        # Rename duplicate genes (not supported by VocabPybind)
        pert_data.adata.var["gene_name"] = (
            pert_data.adata.var["gene_name"]
            .astype(str)
            .where(
                ~pert_data.adata.var["gene_name"].duplicated(),
                pert_data.adata.var["gene_name"].astype(str) + "_dp",
            )
        )

        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + data_params["special_tokens"], None)
        )  # bidirectional lookup [gene <-> int]

    if model_params["model_file"] is not None:
        model_file = model_params["model_file"]
        print("Using pretrained weights from", model_file)

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    data_params["genes"] = {value: index for index, value in enumerate(genes)}
    n_genes = len(genes)

    # Set up scGPT model
    model = TransformerGenerator(
        len(vocab),  # size of vocabulary
        model_params["embsize"],
        model_params["nhead"],
        model_params["d_hid"],
        model_params["nlayers"],
        nlayers_cls=model_params["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_params["dropout"],
        pad_token=data_params["pad_tokens"],
        pad_value=data_params["pad_value"],
        pert_pad_id=data_params["pert_pad_id"],
        do_mvc=trainer_params["mvc"],
        cell_emb_style=trainer_params["cell_emb_style"],
        mvc_decoder_style=trainer_params["mvc_decoder_style"],
        use_fast_transformer=model_params["use_fast_transformer"],
    )
    if model_file is not None:
        print(f"Loading pretrained model from {model_file}")
        pretrained_dict = torch.load(model_file, map_location=device)
        model = load_pretrained(
            model,
            pretrained_dict,
            strict=False,
            prefix=model_params["load_param_prefixs"],
            verbose=True,
        )
    model.to(device)

    if args.load_checkpoint is not None:
        print(f"Loading fine-tuned weights from {args.load_checkpoint}", flush=True)
        state = torch.load(args.load_checkpoint, map_location=device)
        # strict: the checkpoint comes from this same architecture
        model.load_state_dict(state, strict=True)

    # Training
    criterion = None
    if args.train:
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_params["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, trainer_params["schedule_interval"], gamma=0.9
        )
        scaler = torch.cuda.amp.GradScaler(enabled=trainer_params["amp"])
        best_val_loss = float("inf")
        best_model = None
        patience = 0

        for epoch in range(1, trainer_params["epochs"] + 1):
            epoch_start_time = time.time()
            train_loader = pert_data.dataloader["train_loader"]
            valid_loader = pert_data.dataloader["val_loader"]

            # Training epoch
            model.train()
            total_loss = 0.0
            start_time = time.time()

            num_batches = len(train_loader)
            for batch, batch_data in enumerate(train_loader):
                loss = scgpt_forward(
                    batch_data,
                    model,
                    criterion,
                    gene_ids,
                    data_params,
                    trainer_params,
                    n_genes,
                    device,
                )
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                        error_if_nonfinite=False if scaler.is_enabled() else True,
                    )
                    if len(w) > 0:
                        print(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                if batch % trainer_params["log_interval"] == 0 and batch > 0:
                    lr = scheduler.get_last_lr()[0]
                    ms_per_batch = (
                        (time.time() - start_time) * 1000 / trainer_params["log_interval"]
                    )
                    cur_loss = total_loss / trainer_params["log_interval"]
                    print(
                        f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} |"
                    )
                    total_loss = 0
                    start_time = time.time()

            # Validation
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch, batch_data in enumerate(valid_loader):
                    loss = scgpt_forward(
                        batch_data,
                        model,
                        criterion,
                        gene_ids,
                        data_params,
                        trainer_params,
                        n_genes,
                        device,
                    )
                    total_loss += loss.item()
            val_loss = total_loss / len(valid_loader)

            elapsed = time.time() - epoch_start_time
            print("-" * 20)
            print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                  f"valid loss/mse {val_loss:5.4f} |")
            print("-" * 20)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                print(f"Best model with score {best_val_loss:5.4f}")
                patience = 0
            else:
                patience += 1
                if patience >= trainer_params["early_stop"]:
                    print(f"Early stop at epoch {epoch}")
                    break

            scheduler.step()

        # Save the best model
        Path(f"{args.outdir}/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            best_model.state_dict(),
            f"{args.outdir}/checkpoints/{model_name}_seed{args.seed}_{args.dataset}_epoch{epoch}",
        )

        # predict from the best-validation weights
        if best_model is not None:
            model = best_model

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs["split"] == "test"]
    train_adata = pert_data.adata[pert_data.adata.obs["split"] == "train"]

    # Store results
    unique_conds = list(set(test_adata.obs["condition"].unique()) - set(["ctrl"]))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    train_counts = []
    model.eval()
    with torch.no_grad():
        for condition in tqdm(unique_conds):
            gene_list = condition.split("+")
            if "ctrl" in gene_list:
                gene_list.remove("ctrl")

            adata_condition = test_adata[test_adata.obs["condition"] == condition]
            X_post = np.array(adata_condition.X.mean(axis=0))[0]

            # number of this condition's genes seen as single perturbations in training
            n_train = 0
            for g in gene_list:
                if f"{g}+ctrl" in train_adata.obs["condition"].values:
                    n_train += 1
                elif f"ctrl+{g}" in train_adata.obs["condition"].values:
                    n_train += 1
            train_counts.append(n_train)

            # Predict from a pool of control cells
            ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
            if data_params["control_pool_size"] is None:
                data_params["control_pool_size"] = len(ctrl_adata.obs)

            # Some perturbed genes are not measured readout genes; they are still
            # predicted (their perturbation flag simply has no position to set).
            for i in gene_list:
                if i not in pert_data.gene_names.values.tolist():
                    print(f"Warning: {i} is not in the perturbation graph")

            cell_graphs = create_cell_graph_dataset_for_prediction(
                gene_list,
                ctrl_adata,
                pert_data.gene_names.values.tolist(),
                device,
                num_samples=data_params["control_pool_size"],
            )
            loader = DataLoader(
                cell_graphs,
                batch_size=trainer_params["eval_batch_size"],
                shuffle=False,
            )
            preds = []
            for batch_data in loader:
                pred_gene_values = scgpt_forward(
                    batch_data,
                    model,
                    None,
                    gene_ids,
                    data_params,
                    trainer_params,
                    n_genes,
                    device,
                    test=True,
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            preds = np.mean(preds.detach().cpu().numpy(), axis=0)

            post_gt_df.loc[len(post_gt_df)] = X_post
            post_pred_df.loc[len(post_pred_df)] = preds

        index = pd.MultiIndex.from_tuples(
            list(zip(unique_conds, train_counts)), names=["condition", "n_train"]
        )
        post_gt_df.index = index
        post_pred_df.index = index

        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        post_gt_df.to_csv(
            f"{args.outdir}/{args.dataset}_{args.seed}_{model_name}_post-gt.csv"
        )
        post_pred_df.to_csv(
            f"{args.outdir}/{args.dataset}_{args.seed}_{model_name}_post-pred.csv"
        )
