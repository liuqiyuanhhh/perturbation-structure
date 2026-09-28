"""Step 03 -- train PRESAGE (+Perturb-seq) on single-cell data for one CV fold.

Follows upstream's notebooks/PRESAGE_example_k562.ipynb path -- the GEARS
single-cell h5ad straight into ReploglePRESAGEDataModule -- with four runtime
patches (nothing in the PRESAGE checkout is edited):

1. dataloaders run in-process (num_workers=0);
2. read_and_embed allocates one zero matrix per knowledge source
   (patched_read_and_embed.py, not distributed: see README.md);
3. a var index name that collides with a var column is cleared on read;
4. obs gets the "gene" column PRESAGE's Evaluator groups by.

trainer.test() is never called; predictions for the fold's test perturbations
come from trainer.predict with the best-validation checkpoint.  Predictions are
EFFECTS (control-subtracted) on the log1p(CPTT) scale of the single-cell build.

The first fold densifies X into data/<DATASET>/<DATASET>_processed.h5ad and
caches the knowledge tensor; run fold 1 alone before folds 2-5.

Run: PP_DATASET=VCC python 03_train.py --fold 1
"""

import argparse
import datetime
import json
import os
import sys

from config import (
    ATTN_DIR,
    CACHE_DIR,
    CKPT_DIR,
    DATASET,
    DATA_DIR,
    LOG_DIR,
    PRED_DIR,
    PRESAGE_CONFIGS,
    PRESAGE_SRC,
    ROOT,
    RUN,
    WORK,
    current_source_list,
    split_json,
)

# read_and_embed, ModelCheckpoint and CSVLogger use relative paths; keep all of
# them inside the work directory.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(PRESAGE_SRC))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # noqa: E402

import datamodule as datamodule_mod  # noqa: E402
import presage as presage_mod  # noqa: E402
import presage_datamodule as pdm_mod  # noqa: E402
from model_harness import ModelHarness  # noqa: E402
from presage import PRESAGE  # noqa: E402
from presage_datamodule import ReploglePRESAGEDataModule  # noqa: E402
from train import get_predictions, parse_config, set_seed  # noqa: E402

try:
    from patched_read_and_embed import make_patched_read_and_embed  # noqa: E402
except ModuleNotFoundError as error:
    if error.name != "patched_read_and_embed":
        raise
    raise SystemExit(
        "presage_pipeline/patched_read_and_embed.py is missing. It is a modified copy of "
        "PRESAGE's read_and_embed and is not distributed with this repository (Genentech "
        "Non-Commercial Software License). Create it as described in "
        "presage_pipeline/README.md, section 'Required change to PRESAGE'."
    ) from None


# --------------------------------------------------------------------------
# Patch 1: run the dataloaders in-process
# --------------------------------------------------------------------------
# datamodule.py hardcodes num_workers=1, and with
# reload_dataloaders_every_n_epochs=1 Lightning forks a new worker every epoch.
# Forking while OpenMP threads are live can deadlock on CPU; after setup() the
# training tensor is already in memory, so a worker buys nothing.
_OriginalDataLoader = datamodule_mod.DataLoader


def _in_process_dataloader(*args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs.pop("pin_memory", None)
    return _OriginalDataLoader(*args, **kwargs)


datamodule_mod.DataLoader = _in_process_dataloader


# --------------------------------------------------------------------------
# Patch 2: one zero matrix per knowledge source in read_and_embed
# --------------------------------------------------------------------------
presage_mod.read_and_embed = make_patched_read_and_embed(
    presage_mod, cache_dir_override=str(CACHE_DIR)
)


# --------------------------------------------------------------------------
# Patch 3: clear a var index name that collides with a var column
# --------------------------------------------------------------------------
# presage_datamodule.prepare_data only clears var.index.name when there is no
# 'gene_name' column, then calls reset_index(), which fails when both the index
# name and a column are 'gene_name' (as in some GEARS builds).  Clearing the name
# is what upstream does on its other branch; the resulting index is unchanged.
_orig_sc_read = pdm_mod.sc.read


def _sc_read_clear_colliding_var_index(*args, **kwargs):
    adata = _orig_sc_read(*args, **kwargs)
    var = getattr(adata, "var", None)
    if var is not None and var.index.name is not None and var.index.name in var.columns:
        adata.var.index.name = None
    return adata


pdm_mod.sc.read = _sc_read_clear_colliding_var_index


# --------------------------------------------------------------------------
# Patch 4: give obs the "gene" column the Evaluator groups by
# --------------------------------------------------------------------------
# ModelHarness constructs an Evaluator in __init__, which groups
# datamodule.load_preprocessed().obs by "gene".  GEARS builds have no such
# column; the harmonized perturbation label is what it expects.
_orig_load_preprocessed = datamodule_mod.scPerturbDataModule.load_preprocessed


def _load_preprocessed_with_gene(self):
    adata = _orig_load_preprocessed(self)
    if "gene" not in adata.obs.columns:
        field = getattr(self, "perturb_field", "perturbation")
        if field not in adata.obs.columns:
            raise SystemExit(f"preprocessed obs has neither 'gene' nor {field!r}; "
                             f"columns are {list(adata.obs.columns)}")
        adata.obs["gene"] = adata.obs[field].values
    return adata


datamodule_mod.scPerturbDataModule.load_preprocessed = _load_preprocessed_with_gene


# --------------------------------------------------------------------------
def build_config(fold):
    with open(PRESAGE_CONFIGS / "defaults_config.json") as fh:
        config = json.load(fh)  # dotted keys
    with open(PRESAGE_CONFIGS / "singles_config.json") as fh:
        overrides = json.load(fh)  # underscore keys
    with open(ROOT / "configs" / "default_config.json") as fh:
        overrides.update(json.load(fh))

    # underscore -> dotted, on the first underscore only: model_lr -> model.lr
    config.update({
        k.replace("_", ".", 1): v
        for k, v in overrides.items()
        if v is not None and k not in {"config", "data_config"}
    })

    config.update({
        "data.dataset": DATASET,
        "data.data_dir": str(DATA_DIR) + "/",
        "data.seed": str(split_json(fold)),
        "model.pathway_files": str(current_source_list()),
        "training.eval_test": False,
        # singles_config sets the STRING "None"; a real None skips a clustering
        # step whose result is never used.
        "data.nperturb_clusters": None,
    })
    return parse_config(config)


def save_source_weights(module, source_list, fold_tag):
    """Global (gene-independent) knowledge-source weights (the paper's Fig. 4a).

    GATPool keeps the softmaxed global vector on .p_weight_vec after a forward
    pass.  Channel order is [transpose_matrix, coexpression] + the source list.
    """
    pool = getattr(getattr(module, "pool", None), "pool", None)
    vec = getattr(pool, "p_weight_vec", None)
    if vec is None:
        print("  no p_weight_vec on the pool -- skipping source-weight dump")
        return

    weights = vec.detach().cpu().numpy().ravel()
    names = ["transpose_matrix", "coexpression"] + [
        os.path.basename(line) for line in source_list.read_text().split() if line
    ]
    if len(names) != len(weights):
        names = [f"channel_{i}" for i in range(len(weights))]

    ATTN_DIR.mkdir(parents=True, exist_ok=True)
    out = ATTN_DIR / f"{RUN}_{fold_tag}_global_source_weights.json"
    with open(out, "w") as fh:
        json.dump([{"channel": i, "source": n, "weight": float(w)}
                   for i, (n, w) in enumerate(zip(names, weights))], fh, indent=1)
    order = np.argsort(weights)[::-1][:8]
    print(f"\n  top global source weights ({out.name}):")
    for i in order:
        print(f"      {weights[i]:.4f}  {names[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="CV fold, 1-5")
    args = ap.parse_args()
    fold = args.fold
    tag = f"seed_{fold}"

    for d in (PRED_DIR, CKPT_DIR, LOG_DIR, ATTN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    source_list = current_source_list()
    n_sources = len([ln for ln in source_list.read_text().split("\n") if ln.strip()])
    print(f"dataset {DATASET}, fold {fold}")
    print(f"  sources: {source_list.name} ({n_sources} lines -> {n_sources + 2} channels)")

    config = build_config(fold)
    set_seed(config["training"].pop("seed", None))
    config["training"].pop("offline", False)
    config["training"].pop("eval_test", True)
    config["training"].pop("predictions_file", None)
    config["training"].pop("embedding_file", None)
    config["training"].pop("attention_file", None)

    # ---- data ------------------------------------------------------------
    # "perturb_processed.h5ad" is upstream's sentinel for "already on disk".
    ReploglePRESAGEDataModule.urls = dict(
        ReploglePRESAGEDataModule.urls, **{DATASET: "perturb_processed.h5ad"}
    )

    seed_path = config["data"].pop("seed")
    datamodule = ReploglePRESAGEDataModule.from_config(config["data"])
    datamodule.do_test_eval = False
    datamodule.set_seed(seed_path)
    config["data"]["seed"] = seed_path

    datamodule.prepare_data()
    datamodule.setup("fit")

    adata = datamodule.train_dataset.adata
    n_control = int((adata.obs["perturbation"] == datamodule.control_key).sum())
    print(f"\nfold {fold}: train {datamodule.train_dataset.X.shape[0]} perturbations, "
          f"val {datamodule.val_dataset.X.shape[0]}, "
          f"{adata.shape[1]} genes, {adata.shape[0]} training cells "
          f"({n_control} control)")
    if n_control == 0:
        raise SystemExit("no control cells in the training split")

    # ---- model -----------------------------------------------------------
    model_config = config["model"]
    model_config["dataset"] = DATASET
    model_config["pca_dim"] = None          # legacy, unused
    model_config["source"] = "temp"         # legacy, part of the cache key
    model_config["learnable_gene_embedding"] = False

    module = PRESAGE(
        model_config,
        datamodule,
        datamodule.pert_covariates.shape[1],
        datamodule.n_genes,
    )
    if hasattr(module, "custom_init"):
        module.custom_init()

    print(f"\n  knowledge tensor: {tuple(module.gene_embeddings.shape)} "
          f"(genes x {model_config['n_nmf_embedding']} x channels)")

    lightning_module = ModelHarness(module, datamodule, model_config)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(CKPT_DIR),
        filename=f"{RUN}-{tag}-{now}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger(save_dir=str(LOG_DIR), name=RUN, version=tag),
        log_every_n_steps=3,
        num_sanity_val_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, mode="min"),
            checkpoint_callback,
        ],
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=0.1,
        **config["training"],
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    best = checkpoint_callback.best_model_path
    print(f"\nbest checkpoint: {best}")

    # ---- predict ---------------------------------------------------------
    # setup() returns early while _data_setup is True, so reset it first.
    datamodule._data_setup = False
    datamodule.setup("test")
    if datamodule.test_dataset is None:
        raise SystemExit("test_dataset was not built -- setup('test') did not run")

    lightning_module.load_state_dict(
        torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
    )

    predictions = get_predictions(
        trainer, lightning_module, datamodule.test_dataloader(), datamodule.var_names
    )
    predictions = predictions.loc[:, datamodule.train_dataset.adata.var.measured_gene]

    test_perts = datamodule.splits["test"]
    test_predictions = predictions.loc[predictions.index.isin(test_perts)]

    all_path = PRED_DIR / f"{RUN}_{tag}_all_predictions.pkl"
    test_path = PRED_DIR / f"{RUN}_{tag}_test_predictions.pkl"
    joblib.dump(predictions, all_path)
    joblib.dump(test_predictions, test_path)

    print(f"\nwrote {all_path}  {predictions.shape}")
    print(f"wrote {test_path}  {test_predictions.shape}")
    missed = sorted(set(test_perts) - set(test_predictions.index))
    if missed:
        print(f"  NOTE: {len(missed)} test perturbations have no prediction: {missed[:10]}")

    save_source_weights(module, source_list, tag)


if __name__ == "__main__":
    main()
