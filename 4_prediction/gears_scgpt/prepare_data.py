#!/usr/bin/env python3
"""GEARS builds, their 5-fold CV splits and the per-fold reduced builds.

    python prepare_data.py --dataset B [--force]
    python prepare_data.py --parent P --fold F [--force]
    python prepare_data.py --go-graph

``--dataset B`` builds B from its batch-corrected single-cell file(s) and cuts
its five folds:

1. copy each dataset's obs columns to the canonical ``perturbation`` /
   ``batch`` / ``guide_id`` names (``DATASETS[...]['obs_columns']``);
2. keep control cells plus cells whose perturbation passes the perturbation QC;
   optionally subsample the controls (``control_cap``);
3. keep the dataset's primary-outcome gene panel;
4. write GEARS conventions (``condition`` = ``GENE+ctrl`` / ``ctrl``,
   ``cell_type``, ``var['gene_name']``) and run ``PertData.new_data_process``,
   which computes the DE rankings and writes ``perturb_processed.h5ad`` and
   ``data_pyg/``;
5. split the perturbations into 5 test folds, 10% of each fold's training pool
   held out as validation.

The QC tables are the paper tables of 1_preprocessing; they are only read here.
``--parent P --fold F`` builds the reduced build of fold F of a large screen
from P's build.  ``--force`` rebuilds and rewrites the fold files; without it an
existing build or complete set of fold files is left alone.  ``--go-graph``
rebuilds GEARS' GO graph in data/gears from the two pickles there, for machines
without network access.

``run_gears.py`` and ``run_scgpt.py`` load a build under one fold with
``get_pert_data``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from utils.effects import load_quality_audit  # noqa: E402
from utils.qc import load_primary_outcome_lists  # noqa: E402

# GEARS (torch) is imported in the functions that use it, so that run.sh can
# read DATASETS in the analysis environment.

# --- registry ----------------------------------------------------------------
# Every build is named after the screen it comes from.  Its GEARS data directory
# is <GEARS_DATA_DIR>/<name>/<name>/ and the runners address it as <name>-cv5.

DATA_ROOT = str(paths.GEARS_DATA_DIR)
SC_DIR = str(paths.SC_DIR)
OUTCOME_QC = paths.OUTCOME_QC
PERT_QC = paths.PERT_QC

# h5ad        file(s) under paths.SC_DIR.  A list is concatenated over cells with
#             the genes intersected (Feng-gw ships as two halves of one screen).
# qc_dataset  the screen's label inside the QC tables.
# cell_type   written to obs['cell_type']; GEARS uses it as the DE covariate.
# obs_columns {source obs column: canonical name}.  The h5ads do not share one
#             obs schema; only canonical columns that are missing get filled.
# control_cap keep a batch-stratified random sample of this many control cells.
NADIG_OBS = {'gene': 'perturbation', 'gem_group': 'batch', 'sgID_AB': 'guide_id'}
FENG_OBS = {'Batch': 'batch', 'Guide_Call': 'guide_id'}
HUANG_OBS = {'gene_target': 'perturbation', 'sample': 'batch', 'guide_target': 'guide_id'}

DATASETS = {
    'VCC': {
        'h5ad': 'VCC.h5ad', 'qc_dataset': 'VCC', 'cell_type': 'hESC',
    },
    'Replogle-E-k562': {
        'h5ad': 'Replogle-E-k562.h5ad', 'qc_dataset': 'Replogle-E-k562',
        'cell_type': 'K562',
    },
    'Replogle-E-rpe1': {
        'h5ad': 'Replogle-E-rpe1.h5ad', 'qc_dataset': 'Replogle-E-rpe1',
        'cell_type': 'RPE1',
    },
    'Replogle-GW-k562': {
        'h5ad': 'Replogle-GW-k562.h5ad', 'qc_dataset': 'Replogle-GW-k562',
        'cell_type': 'K562',
    },
    'Nadig-HEPG2': {
        'h5ad': 'Nadig-HEPG2.h5ad', 'qc_dataset': 'Nadig-HEPG2',
        'cell_type': 'HEPG2', 'obs_columns': NADIG_OBS,
    },
    'Nadig-JURKAT': {
        'h5ad': 'Nadig-JURKAT.h5ad', 'qc_dataset': 'Nadig-JURKAT',
        'cell_type': 'JURKAT', 'obs_columns': NADIG_OBS,
    },
    'Feng-ts': {
        'h5ad': 'Feng-ts.h5ad', 'qc_dataset': 'Feng-ts',
        'cell_type': 'Feng-ts', 'obs_columns': FENG_OBS,
    },
    # Feng-gwsf (fitness-gene pool) and Feng-gwsnf (non-fitness pool) are two
    # halves of one genome-wide screen: disjoint cells and perturbations, shared
    # 'control' label.  The QC tables were computed on their gene intersection.
    'Feng-gw': {
        'h5ad': ['Feng-gwsf.h5ad', 'Feng-gwsnf.h5ad'], 'qc_dataset': 'Feng-GW',
        'cell_type': 'iPSC', 'obs_columns': FENG_OBS,
    },
    # Feng-gw with 50000 of its ~500000 control cells kept.  Every QC-passing
    # perturbation keeps all of its cells, so the fold files are identical to
    # Feng-gw's; GEARS and scGPT-ft train on this build.
    'Feng-gw-control-cap': {
        'h5ad': ['Feng-gwsf.h5ad', 'Feng-gwsnf.h5ad'], 'qc_dataset': 'Feng-GW',
        'cell_type': 'iPSC', 'obs_columns': FENG_OBS, 'control_cap': 50_000,
    },
    'Pan-GW-hESC': {
        'h5ad': 'Pan-GW-hESC.h5ad', 'qc_dataset': 'Pan-GW-hESC',
        'cell_type': 'hESC',
    },
    'Nourreddine-GW-ipsc': {
        'h5ad': 'Nourreddine-GW-ipsc.h5ad', 'qc_dataset': 'Nourreddine-GW-ipsc',
        'cell_type': 'iPSC',
    },
    'Huang-HCT116': {
        'h5ad': 'Huang-HCT116.h5ad', 'qc_dataset': 'Huang-HCT116',
        'cell_type': 'HCT116', 'obs_columns': HUANG_OBS,
    },
    'Huang-HEK293T': {
        'h5ad': 'Huang-HEK293T.h5ad', 'qc_dataset': 'Huang-HEK293T',
        'cell_type': 'HEK293T', 'obs_columns': HUANG_OBS,
    },
}

# Per-fold reduced builds (build_fold_subset).  For the three largest screens
# GEARS and scGPT-ft train on one reduced build per fold: the fold's test
# perturbations with all of their cells, a third of its train and val
# perturbations, and FOLD_SUBSET_CONTROL_CAP control cells.  A fold needs its own
# build because the five test sets are disjoint, so what fold 1 may drop is fold
# 2's test data.  They are addressed as '<parent>-third-f<fold>-cv5'.
FOLD_SUBSET_PARENTS = ('Huang-HCT116', 'Huang-HEK293T', 'Nourreddine-GW-ipsc')
FOLD_SUBSET_FRACTION = 1.0 / 3.0
FOLD_SUBSET_CONTROL_CAP = 50_000
FOLD_SUBSET_SEED = 0


def fold_subset_name(parent: str, fold: int) -> str:
    return f'{parent}-third-f{fold}'


for _parent in FOLD_SUBSET_PARENTS:
    for _fold in range(1, 6):
        DATASETS[fold_subset_name(_parent, _fold)] = dict(
            DATASETS[_parent], fold_subset_of=_parent, fold=_fold)
del _parent, _fold

# Anything in obs['perturbation'] matching one of these (case-insensitive) is a
# control cell: kept unconditionally, and mapped to the 'ctrl' condition.
CONTROL_LABELS = {'control', 'ctrl', 'non-targeting', 'non_targeting', 'nt', 'ntc'}

# obs columns carried into perturb_processed.h5ad.  GEARS casts all of obs to
# category, which is slow on the continuous per-cell QC columns these files ship.
KEEP_OBS = ['perturbation', 'batch', 'guide_id']

# Cross-validation
N_FOLDS = 5
VAL_FRACTION = 0.1      # of the 4-fold training pool
CONTROL_CAP_SEED = 0    # RNG for the control subsample (see control_cap)
FOLD_SEED = 0           # RNG for the fold partition
VAL_SEED_BASE = 100     # RNG for fold f's val carve-out: VAL_SEED_BASE + f


def build_path(name: str) -> str:
    """Directory holding the prepared GEARS build of ``name``."""
    return f'{DATA_ROOT}/{name}/{name}'


def dataset_dir(name: str) -> str:
    """PertData root for ``name`` (its parent; PertData writes name/ inside)."""
    return f'{DATA_ROOT}/{name}'


def split_file(name: str, fold: int) -> str:
    return f'{build_path(name)}/splits/{name}_cv5_{fold}.pkl'


# --- GEARS support files -----------------------------------------------------
# PertData.__init__ fetches gene2go_all.pkl from Harvard Dataverse, and
# prepare_split fetches essential_all_data_pert_genes.pkl and the
# go_essential_all GO graph.  On a node without outbound network the download
# writes a zero-byte file and the next pickle.load fails with EOFError.  GEARS
# skips a download when the target already exists, so copying these in from
# paths.GEARS_SUPPORT_DIR or from another build is enough.  The GO directory is
# symlinked rather than copied (~355 MB, read-only).  If only the two pickles
# are at hand, --go-graph rebuilds the GO graph from them.

PICKLES = ['gene2go_all.pkl', 'essential_all_data_pert_genes.pkl']
GO_DIR = 'go_essential_all'
GO_CSV = 'go_essential_all.csv'


def reference_dirs() -> list[Path]:
    """Directories that may already hold a good copy of a support file.

    Discovered at call time: PertData downloads into a dataset's parent directory
    during a build, and the staging call for the inner directory has to be able to
    see what has just appeared next to it.
    """
    root = Path(DATA_ROOT)
    dirs = [paths.GEARS_SUPPORT_DIR, root]
    for child in sorted(root.glob('*')) if root.is_dir() else []:
        if not child.is_dir():
            continue
        dirs.append(child)
        inner = child / child.name      # PertData writes <root>/<name>/<name>/
        if inner.is_dir():
            dirs.append(inner)
    return dirs


def _usable(path: Path) -> bool:
    """Zero-byte files and broken symlinks are failed downloads, not local copies."""
    if path.is_symlink() and not path.exists():
        return False
    if path.is_dir():
        csv = path / GO_CSV
        if csv.exists():
            return csv.stat().st_size > 0
        return any(path.iterdir())
    return path.exists() and path.stat().st_size > 0


def _find(name: str, extra_references: list[Path] | None = None) -> Path | None:
    for reference in list(extra_references or []) + reference_dirs():
        candidate = Path(reference) / name
        if _usable(candidate):
            return candidate
    return None


def build_go_edge_list(support_dir: Path, threshold: float = 0.1,
                       block: int = 512) -> pd.DataFrame:
    """The go_essential_all table from gene2go_all.pkl and essential_all_data_pert_genes.pkl.

    gene2go_all.pkl maps gene -> set of GO terms; essential_all_data_pert_genes.pkl
    is the gene universe GEARS uses as its perturbable gene list.  The rule is the
    one GEARS applies in ``utils.get_GO_edge_list``:

        score(g1, g2) = |GO(g1) & GO(g2)| / |GO(g1) | GO(g2)|,  kept when > 0.1

    Self-pairs are kept (score 1.0), as in that function.  Columns are
    ``source,target,importance``, which is what ``get_similarity_network`` reads.
    The Jaccard is computed with sparse matrix products rather than GEARS' double
    loop.
    """
    with open(support_dir / 'gene2go_all.pkl', 'rb') as handle:
        gene2go = pickle.load(handle)
    with open(support_dir / 'essential_all_data_pert_genes.pkl', 'rb') as handle:
        essential = pickle.load(handle)

    genes = [str(g) for g in essential]
    genes = [g for g in dict.fromkeys(genes) if g in gene2go]   # dedup, keep order
    print(f'{len(genes)} essential genes carry GO annotation')

    terms = sorted({term for gene in genes for term in gene2go[gene]})
    term_index = {term: i for i, term in enumerate(terms)}
    rows, cols = [], []
    for row, gene in enumerate(genes):
        for term in gene2go[gene]:
            rows.append(row)
            cols.append(term_index[term])
    membership = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(genes), len(terms)),
    )
    sizes = np.asarray(membership.sum(axis=1)).ravel()
    print(f'{len(terms)} distinct GO terms, {membership.nnz} annotations')

    gene_array = np.asarray(genes, dtype=object)
    frames = []
    for start in range(0, len(genes), block):
        stop = min(start + block, len(genes))
        intersection = np.asarray(
            (membership[start:stop] @ membership.T).todense(), dtype=np.float32
        )
        union = sizes[start:stop, None] + sizes[None, :] - intersection
        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.where(union > 0, intersection / union, 0.0)
        keep = np.argwhere(score > threshold)
        if keep.size:
            frames.append(pd.DataFrame({
                'source': gene_array[keep[:, 0] + start],
                'target': gene_array[keep[:, 1]],
                'importance': score[keep[:, 0], keep[:, 1]].astype(np.float64),
            }))

    edges = pd.concat(frames, ignore_index=True)
    print(f'{len(edges)} edges with jaccard > {threshold}')
    return edges


def stage_support_files(target_dir, extra_references=None):
    """Populate ``target_dir`` with the files GEARS expects to download.

    Copies the two pickles and symlinks the GO graph directory.  Existing good
    copies are left alone; zero-byte leftovers from a failed download are
    replaced.  Returns the names that could not be sourced locally (GEARS will
    then try to download them).
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    references = [Path(p) for p in (extra_references or [])]
    missing = []

    for name in PICKLES:
        destination = target / name
        if _usable(destination):
            print(f'  [support] {name}: already present')
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()          # zero-byte failed download
        source = _find(name, references)
        if source is None:
            missing.append(name)
            print(f'  [support] {name}: not found locally; GEARS will download it')
            continue
        shutil.copy2(source, destination)
        print(f'  [support] {name}: copied')

    destination = target / GO_DIR
    if _usable(destination):
        print(f'  [support] {GO_DIR}/: already present')
    else:
        if destination.is_symlink() or destination.exists():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()     # broken symlink or empty dir
        source = _find(GO_DIR, references)
        if source is None:
            missing.append(GO_DIR)
            print(f'  [support] {GO_DIR}/: not found locally; GEARS will download it '
                  '(or rebuild it offline with prepare_data.py --go-graph)')
        else:
            os.symlink(source.resolve(), destination)
            print(f'  [support] {GO_DIR}/: linked')

    # a stale zero-byte tarball makes tarfile.open raise instead of re-downloading
    stale_tar = target / f'{GO_DIR}.tar.gz'
    if stale_tar.exists() and stale_tar.stat().st_size == 0:
        stale_tar.unlink()

    return missing


# --- build -------------------------------------------------------------------
# The input h5ads already hold batch-corrected log1p(CPTT) values, so nothing is
# normalised here: normalize_total / log1p would rescale each cell by a total
# that is no longer a UMI count and log a second time.

def value_stats(matrix) -> dict:
    """Sampled X values for the QC report (raw counts would be integer-valued)."""
    sample = matrix.data[:2_000_000] if issparse(matrix) else np.asarray(matrix).ravel()[:2_000_000]
    sample = sample[np.isfinite(sample)]
    return {
        'sampled_values': int(sample.size),
        'fraction_integer_valued': float(np.mean(np.isclose(sample, np.round(sample)))),
        'min': float(sample.min()),
        'max': float(sample.max()),
    }


def canonicalise_obs(adata, spec: dict) -> dict:
    """Copy this dataset's obs columns to the canonical names.

    Only missing canonical columns are filled, so a mapping listed for a dataset
    that already has the column is a no-op (Replogle-E-k562 ships both
    'perturbation' and 'gene', which differ in their control label).
    """
    applied = {}
    for source, canonical in (spec.get('obs_columns') or {}).items():
        if canonical in adata.obs.columns:
            continue
        column = adata.obs[source]
        # batch / guide are provenance labels, not numbers
        if canonical != 'perturbation' and column.dtype.kind in 'iuf':
            column = column.astype(str)
        adata.obs[canonical] = column.values
        applied[source] = canonical
    if applied:
        print('  obs columns mapped: '
              + ', '.join(f'{k} -> {v}' for k, v in applied.items()), flush=True)
    return applied


def read_sources(spec: dict, name: str):
    """Read the dataset's h5ad(s), concatenating cells when there is more than one.

    Genes are intersected (anndata's join='inner'), which is the gene universe the
    QC tables were computed on for a multi-file screen.
    """
    sources = spec['h5ad']
    if isinstance(sources, str):
        sources = [sources]
    sc_paths = [f'{SC_DIR}/{f}' for f in sources]
    if len(sc_paths) == 1:
        print(f'[{name}] reading {sources[0]}', flush=True)
        return anndata.read_h5ad(sc_paths[0]), sc_paths

    parts = []
    for source, path in zip(sources, sc_paths):
        print(f'[{name}] reading {source}', flush=True)
        part = anndata.read_h5ad(path)
        print(f'    {part.shape[0]} cells x {part.shape[1]} genes', flush=True)
        parts.append(part)

    merged = anndata.concat(parts, axis=0, join='inner', index_unique=None)
    del parts
    print(f'  merged: {merged.shape[0]} cells x {merged.shape[1]} genes', flush=True)
    return merged, sc_paths


def cap_controls(is_control, keep_cells, strata, cap):
    """Thin the control cells down to ``cap``, keeping every stratum's share.

    Returns the number of controls kept; ``keep_cells`` is modified in place.
    The sample is stratified by ``strata`` (the batch column) with a
    largest-remainder allocation, so every batch keeps its share of controls.
    """
    ctrl_idx = np.flatnonzero(is_control & keep_cells)
    if ctrl_idx.size <= cap:
        print(f'  control cap {cap}: {ctrl_idx.size} controls, nothing to drop',
              flush=True)
        return int(ctrl_idx.size)

    rng = np.random.default_rng(CONTROL_CAP_SEED)
    _, inverse = np.unique(strata[ctrl_idx], return_inverse=True)
    counts = np.bincount(inverse)
    exact = counts * (cap / counts.sum())
    take = np.minimum(np.floor(exact).astype(int), counts)

    # Largest remainder, repeated: one pass can still come up short when several
    # strata are pinned at their own size by the min() above.
    while take.sum() < cap:
        room = take < counts
        order = np.argsort(-(exact - take))
        for j in order[room[order]]:
            take[j] += 1
            if take.sum() == cap:
                break

    chosen = np.concatenate([
        ctrl_idx[inverse == j] if take[j] >= counts[j]
        else rng.choice(ctrl_idx[inverse == j], size=take[j], replace=False)
        for j in range(counts.size)
    ])
    keep_cells[np.setdiff1d(ctrl_idx, chosen, assume_unique=False)] = False
    print(f'  control cap {cap}: {ctrl_idx.size} -> {chosen.size} control cells '
          f'across {counts.size} strata (seed {CONTROL_CAP_SEED})', flush=True)
    return int(chosen.size)


def build(name: str, force: bool) -> None:
    from gears import PertData

    spec = DATASETS[name]
    out_dir = dataset_dir(name)
    data_path = build_path(name)
    processed = f'{data_path}/perturb_processed.h5ad'
    report_path = f'{out_dir}/{name}_qc_report.json'

    if Path(processed).exists() and not force:
        print(f'[{name}] already built -- nothing to do (pass --force to rebuild)')
        return

    adata, sc_paths = read_sources(spec, name)
    n_cells_raw, n_genes_raw = adata.shape
    print(f'  raw: {n_cells_raw} cells x {n_genes_raw} genes', flush=True)

    stats = value_stats(adata.X)
    print('  X: {fraction_integer_valued:.4f} of sampled values are integers, '
          'range [{min:.4f}, {max:.4f}]'.format(**stats), flush=True)

    obs_mapped = canonicalise_obs(adata, spec)

    # --- QC lists
    qc_dataset = spec['qc_dataset']
    panel = load_primary_outcome_lists(OUTCOME_QC)[qc_dataset]
    _, passing_by_dataset = load_quality_audit(PERT_QC)
    passing = passing_by_dataset[qc_dataset]
    print(f'  QC[{qc_dataset}]: {len(panel)} primary outcome genes, '
          f'{len(passing)} passing perturbations', flush=True)

    # --- cells: controls plus QC-passing perturbations
    labels = adata.obs['perturbation'].astype(str).values
    is_control = np.asarray([x.strip().lower() in CONTROL_LABELS for x in labels])
    keep_cells = is_control | np.isin(labels, list(passing))
    measured = sorted(set(labels[~is_control]))
    dropped_perts = sorted(set(labels[~keep_cells]))   # before the control cap
    n_control_available = int(is_control.sum())
    print(f'  cells: {n_cells_raw} -> {int(keep_cells.sum())} '
          f'({n_control_available} control); perturbations {len(measured)} -> '
          f'{len(measured) - len(dropped_perts)} ({len(dropped_perts)} failed QC)', flush=True)

    control_cap = spec.get('control_cap')
    if control_cap is None:
        n_control_kept = n_control_available
    else:
        strata = (adata.obs['batch'].astype(str).values
                  if 'batch' in adata.obs.columns
                  else np.zeros(adata.n_obs, dtype='U1'))
        n_control_kept = cap_controls(is_control, keep_cells, strata, control_cap)

    # --- genes: the QC primary-outcome panel, in var order
    panel_set = set(panel)
    keep_genes = np.asarray(adata.var_names.isin(panel_set))
    missing_panel = sorted(panel_set - set(adata.var_names))
    if missing_panel:
        print(f'  [WARN] {len(missing_panel)} panel genes absent from var, '
              f'e.g. {missing_panel[:5]}', flush=True)
    print(f'  genes: {n_genes_raw} -> {int(keep_genes.sum())}', flush=True)

    kept_labels = {x for x in labels[keep_cells]
                   if x.strip().lower() not in CONTROL_LABELS}
    off_panel_targets = sorted(kept_labels - set(adata.var_names[keep_genes]))
    if off_panel_targets:
        print(f'  [WARN] {len(off_panel_targets)} retained perturbations target a gene '
              f'outside the panel, e.g. {off_panel_targets[:5]}', flush=True)

    # --- subset once (one copy, rather than one per axis)
    adata = adata[keep_cells, keep_genes].copy()
    n_cells_qc, n_genes_qc = adata.shape
    print(f'  after QC: {n_cells_qc} cells x {n_genes_qc} genes', flush=True)

    # --- GEARS conventions
    obs_keep = [c for c in KEEP_OBS if c in adata.obs.columns]
    adata.obs = adata.obs[obs_keep].copy()
    condition = adata.obs['perturbation'].astype(str)
    is_ctrl = condition.str.strip().str.lower().isin(CONTROL_LABELS)
    adata.obs['condition'] = np.where(is_ctrl, 'ctrl', condition + '+ctrl')
    adata.obs['cell_type'] = spec['cell_type']
    adata.var['gene_name'] = adata.var_names
    adata.X = csr_matrix(adata.X)      # no-op when it is already csr

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as handle:
        json.dump({
            'dataset': name,
            'source_h5ad': sc_paths if len(sc_paths) > 1 else sc_paths[0],
            'qc_dataset': qc_dataset,
            'value_stats': stats,
            'n_cells_raw': int(n_cells_raw), 'n_cells_qc': int(n_cells_qc),
            'n_genes_raw': int(n_genes_raw), 'n_genes_qc': int(n_genes_qc),
            'n_control_cells': n_control_kept,
            'n_control_cells_available': n_control_available,
            'control_cap': control_cap,
            'control_cap_seed': CONTROL_CAP_SEED if control_cap else None,
            'n_perturbations_measured': len(measured),
            'n_perturbations_dropped': len(dropped_perts),
            'perturbations_dropped': dropped_perts,
            'n_panel_genes_missing_from_sc': len(missing_panel),
            'obs_columns_kept': obs_keep + ['condition', 'cell_type'],
            'obs_columns_mapped': obs_mapped,
        }, handle, indent=2)

    stage_support_files(out_dir)

    pert_data = PertData(out_dir)
    pert_data.new_data_process(dataset_name=name, adata=adata)
    if name != name.lower():  # cell-gears writes the build under the lower-cased name
        shutil.rmtree(data_path, ignore_errors=True)  # an older build, with --force
        os.replace(os.path.join(out_dir, name.lower()), data_path)
    print('  new_data_process done', flush=True)

    # prepare_split() reads the support files from the build directory at
    # training time; PertData has just downloaded them one level up.
    stage_support_files(data_path, extra_references=[out_dir])


# --- 5-fold splits -----------------------------------------------------------
# Fold f is the test set of run f; the other four folds are the training pool,
# of which 10% is held out as validation.  Every perturbation is tested exactly
# once, so the five prediction files concatenate into one complete evaluation
# set.  Output is GEARS' split='custom' format: a pickled
# {'train': [...], 'val': [...], 'test': [...]} of condition names at
# split_file(name, fold), plus a manifest of every fold's test set.  'ctrl' is
# appended to train, which the custom branch of prepare_split does not do by
# itself.  Conditions whose target gene is outside the GEARS perturbation graph
# are dropped first, exactly as PertData.load (filter_pert_in_go) would drop
# their cells.

def read_conditions(h5ad: Path) -> list[str]:
    """Values of ``obs['condition']`` that are used by at least one cell."""
    with h5py.File(h5ad, 'r') as handle:
        node = handle['obs']['condition']
        if isinstance(node, h5py.Group):            # categorical encoding
            categories = np.asarray(node['categories'].asstr()[:])
            codes = node['codes'][:]
            used = np.unique(codes)
            values = categories[used[used >= 0]]    # -1 marks NaN
        else:
            values = np.unique(node.asstr()[:])
    return sorted(str(v) for v in values)


def pert_names(data_path: Path) -> set[str]:
    """Reproduce ``PertData.set_pert_genes()``: the genes GEARS can perturb."""
    with open(data_path / 'essential_all_data_pert_genes.pkl', 'rb') as handle:
        essential = pickle.load(handle)
    with open(data_path / 'gene2go_all.pkl', 'rb') as handle:
        gene2go = pickle.load(handle)
    return {gene for gene in essential if gene in gene2go}


def keep_condition(condition: str, names: set[str]) -> bool:
    """``gears.utils.filter_pert_in_go``, for single- and two-gene conditions."""
    if condition == 'ctrl':
        return True
    parts = condition.split('+')
    if len(parts) != 2:
        return False
    n_ctrl = sum(part == 'ctrl' for part in parts)
    n_known = sum(part in names for part in parts)
    return n_ctrl + n_known == 2


def build_splits(name: str, force: bool) -> None:
    data_path = Path(build_path(name))
    if not force and all(Path(split_file(name, f)).exists() for f in range(1, N_FOLDS + 1)):
        print(f'[{name}] fold files exist -- nothing to do (pass --force to rewrite)')
        return
    print(f'[{name}] {N_FOLDS}-fold splits', flush=True)

    conditions = read_conditions(data_path / 'perturb_processed.h5ad')
    names = pert_names(data_path)
    kept = [c for c in conditions if keep_condition(c, names)]
    dropped = sorted(set(conditions) - set(kept))
    if dropped:
        print(f'  {len(dropped)} condition(s) outside the GEARS perturbation graph, '
              f'dropped as PertData.load would: {dropped[:8]}'
              f'{" ..." if len(dropped) > 8 else ""}', flush=True)
    perts = [c for c in kept if c != 'ctrl']

    rng = np.random.RandomState(FOLD_SEED)
    order = rng.permutation(len(perts))
    folds = [np.asarray(part) for part in np.array_split(order, N_FOLDS)]

    split_dir = data_path / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)

    written, fold_members = [], {}
    for fold, test_idx in enumerate(folds, start=1):
        test = [perts[i] for i in test_idx]
        pool = np.concatenate([g for j, g in enumerate(folds) if j != fold - 1])

        rng = np.random.RandomState(VAL_SEED_BASE + fold)
        pool = rng.permutation(pool)
        n_val = int(len(pool) * VAL_FRACTION)
        val = [perts[i] for i in pool[:n_val]]
        train = [perts[i] for i in pool[n_val:]]
        train.append('ctrl')

        out = Path(split_file(name, fold))
        with open(out, 'wb') as handle:
            pickle.dump({'train': train, 'val': val, 'test': test}, handle)

        written.append(out.name)
        fold_members[fold] = sorted(test)
        print(f'  fold {fold}: train {len(train):5d} (incl ctrl)  val {len(val):4d}  '
              f'test {len(test):5d}  -> {out.name}', flush=True)

    manifest = {
        'dataset': name,
        'n_perturbations': len(perts),
        'n_folds': N_FOLDS,
        'val_fraction_of_pool': VAL_FRACTION,
        'fold_seed': FOLD_SEED,
        'val_seed_base': VAL_SEED_BASE,
        'split_files': written,
        'test_sets': fold_members,
    }
    with open(split_dir / f'{name}_cv5_manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2)


# --- per-fold reduced builds -------------------------------------------------
# For Huang-HCT116, Huang-HEK293T and Nourreddine-GW-ipsc a full GEARS build does
# not fit in memory for training (GEARS unpickles every cell graph up front).
# Each fold therefore gets its own, smaller build:
#
#     cells kept = the fold's test perturbations (ALL of their cells)
#                + a third of its train perturbations
#                + a third of its val perturbations
#                + FOLD_SUBSET_CONTROL_CAP control cells, stratified by batch
#
# The test set is untouched, so evaluation windows are the same as for the
# parent build; only what the models learn from shrinks.  The source is the
# parent's perturb_processed.h5ad, so QC, gene panel and DE rankings are
# inherited unchanged -- every kept condition keeps all of its own cells, so its
# DE ranking against control means what it meant in the parent.

CTRL = 'ctrl'


def reduce_one(conditions, fraction, rng):
    """Keep ``fraction`` of the conditions; 'ctrl' is always kept."""
    keep_always = [c for c in conditions if str(c).strip().lower() == CTRL]
    pool = [c for c in conditions if str(c).strip().lower() != CTRL]
    n_keep = int(round(len(pool) * fraction))
    chosen = rng.choice(np.asarray(pool, dtype=object), size=n_keep, replace=False)
    chosen = sorted(chosen.tolist())
    dropped = sorted(set(pool) - set(chosen))
    return keep_always + chosen, dropped


def build_fold_subset(parent: str, fold: int, force: bool) -> None:
    from gears import PertData

    name = fold_subset_name(parent, fold)
    out_dir = Path(dataset_dir(name))
    data_path = Path(build_path(name))
    processed = data_path / 'perturb_processed.h5ad'
    pyg = data_path / 'data_pyg' / 'cell_graphs.pkl'
    if pyg.exists() and not force:
        print(f'[{name}] already built -- nothing to do (pass --force to rebuild)')
        return

    src = Path(build_path(parent))

    # --- which conditions survive
    with open(split_file(parent, fold), 'rb') as handle:
        split = pickle.load(handle)
    rng = np.random.default_rng(FOLD_SUBSET_SEED + fold)
    train, _ = reduce_one(split['train'], FOLD_SUBSET_FRACTION, rng)
    val, _ = reduce_one(split['val'], FOLD_SUBSET_FRACTION, rng)
    test = list(split['test'])
    keep_conditions = set(train) | set(val) | set(test)
    print(f'  fold {fold}: train {len(split["train"])} -> {len(train)}, '
          f'val {len(split["val"])} -> {len(val)}, test {len(test)} (unchanged)',
          flush=True)

    # --- cells.  Backed read, so only the kept rows are ever materialised.
    adata = anndata.read_h5ad(src / 'perturb_processed.h5ad', backed='r')
    n_cells_parent = adata.n_obs
    condition = adata.obs['condition'].astype(str).values
    keep_cells = np.isin(condition, list(keep_conditions))
    is_control = condition == CTRL
    n_control_parent = int(is_control.sum())

    strata = (adata.obs['batch'].astype(str).values
              if 'batch' in adata.obs.columns
              else np.zeros(adata.n_obs, dtype='U1'))
    n_control_kept = cap_controls(is_control, keep_cells, strata, FOLD_SUBSET_CONTROL_CAP)

    print(f'  cells: {n_cells_parent} -> {int(keep_cells.sum())} '
          f'(control {n_control_parent} -> {n_control_kept})', flush=True)
    adata = adata[keep_cells].to_memory()
    adata.X = csr_matrix(adata.X)

    # --- write the build.  PertData.load fills in data_pyg below.
    data_path.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed)
    print(f'  wrote perturb_processed.h5ad ({processed.stat().st_size / 1e9:.1f} GB)',
          flush=True)

    (data_path / 'splits').mkdir(exist_ok=True)
    with open(split_file(name, fold), 'wb') as handle:
        pickle.dump({'train': train, 'val': val, 'test': test}, handle)

    with open(out_dir / f'{name}_build_report.json', 'w') as handle:
        json.dump({'name': name, 'parent': parent, 'fold': fold,
                   'fraction': FOLD_SUBSET_FRACTION, 'seed': FOLD_SUBSET_SEED,
                   'control_cap': FOLD_SUBSET_CONTROL_CAP,
                   'n_cells_parent': int(n_cells_parent),
                   'n_cells': int(adata.n_obs),
                   'n_genes': int(adata.n_vars),
                   'n_control_parent': n_control_parent,
                   'n_control': n_control_kept,
                   'n_train': len(train), 'n_val': len(val), 'n_test': len(test)},
                  handle, indent=2)

    stage_support_files(out_dir, extra_references=[data_path, src])
    stage_support_files(data_path, extra_references=[out_dir, src])

    print('  PertData.load -> building data_pyg ...', flush=True)
    pert_data = PertData(str(data_path))
    pert_data.load(data_name=name, data_path=str(data_path))
    print(f'  done: cell_graphs.pkl ({pyg.stat().st_size / 1e9:.1f} GB), '
          f'{len(pert_data.dataset_processed)} conditions', flush=True)


# --- loader for run_gears.py and run_scgpt.py --------------------------------
# Both runners call get_pert_data(dataset, seed) with dataset = <build>-cv5 and
# seed = the FOLD INDEX 1..5 (not a random seed).  The five test sets are
# disjoint and together cover every perturbation exactly once.

# dataset name passed to the runners -> the prepared build it loads
CV5_DATASETS = {f'{name}-cv5': name for name in DATASETS}


def apply_set2conditions_to_obs(pert_data, split_name: str = 'split'):
    """Write ``pert_data.set2conditions`` back onto ``adata.obs[split_name]``.

    ``prepare_split(split='custom')`` loads the split dict but never touches
    ``adata.obs``, while both runners select their evaluation cells with
    ``adata.obs['split'] == 'test'``.
    """
    mapping = {}
    for split, conditions in pert_data.set2conditions.items():
        for condition in conditions:
            mapping[condition] = split

    obs = pert_data.adata.obs
    obs[split_name] = obs['condition'].astype(str).map(mapping).to_numpy()
    return pert_data


def cv5_pre(dataset: str, fold: int, data_dir: str | None = None):
    """Load a prepared build under one 5-fold CV split (fold 1..5)."""
    from gears import PertData

    name = CV5_DATASETS[dataset]
    root = data_dir or DATA_ROOT
    data_path = f'{root}/{name}/{name}'
    split_path = f'{data_path}/splits/{name}_cv5_{fold}.pkl'

    pert_data = PertData(data_path)
    pert_data.load(data_name=name, data_path=data_path)
    # seed=fold keeps GEARS' co-expression network cache (refit on the training
    # cells) separate per fold, so no fold's graph sees its own test cells.
    pert_data.prepare_split(split='custom', seed=fold, split_dict_path=split_path)
    apply_set2conditions_to_obs(pert_data)
    pert_data.dataset_name = ''
    return pert_data


def get_pert_data(dataset: str, seed: int, data_dir: str | None = None, **kwargs):
    """Entry point the runners call.  ``seed`` is the fold index 1..5."""
    return cv5_pre(dataset, fold=int(seed), data_dir=data_dir, **kwargs)


def main() -> None:
    buildable = sorted(k for k, v in DATASETS.items() if 'fold_subset_of' not in v)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dataset', choices=buildable + ['all'],
                      help='build B and write its fold files')
    mode.add_argument('--parent', choices=FOLD_SUBSET_PARENTS,
                      help='with --fold: the reduced build of one fold of P')
    mode.add_argument('--go-graph', action='store_true',
                      help='rebuild data/gears/go_essential_all/ from the two pickles '
                           'in data/gears (no network needed)')
    parser.add_argument('--fold', type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument('--force', action='store_true',
                        help='rebuild, and rewrite the fold files, even if they exist')
    args = parser.parse_args()

    if args.go_graph:
        out_dir = paths.GEARS_SUPPORT_DIR / GO_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        build_go_edge_list(paths.GEARS_SUPPORT_DIR).to_csv(out_dir / GO_CSV, index=False)
        return
    if args.parent:
        build_fold_subset(args.parent, args.fold, args.force)
        return
    names = buildable if args.dataset == 'all' else [args.dataset]
    for name in names:
        build(name, args.force)
        build_splits(name, args.force)
    print('DONE')


if __name__ == '__main__':
    main()
