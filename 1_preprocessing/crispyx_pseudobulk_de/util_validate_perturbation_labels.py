"""Check perturbation labels for truncation defects before QC (step 03).

A naive first-underscore split of guide names merges distinct genes whose symbol
contains a hyphen (e.g. HLA-B). validate_dataset reads obs/var only and:

1. (fails) where a guide column exists, recovers each cell's target by stripping a
   trailing 18-25 nt spacer and mapping '_' to '-', and flags labels inconsistent
   with it (datasets using another guide-naming scheme report INCONCLUSIVE);
2. (warns) more than 500 perturbations without any hyphenated label;
3. (warns) a label absent from var while var contains "<label>-*";
4. (info) number of labels absent from var.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

SPACER = re.compile(r"_[ACGT]{18,25}$")
GUIDE_COL_CANDIDATES = [
    "Guide_Call", "gRNA", "guide_id", "guide", "sgRNA", "sgrna",
    "guide_target_raw", "guide_target", "sgrna_name", "target_gene",
    "gene_target",
]
CONTROL_SYNONYMS = {"control", "ntc", "non-targeting", "nontargeting", "unassigned"}
ZERO_HYPHEN_MIN_LABELS = 500


def _is_control_synonym(s: str) -> bool:
    return s.lower() in CONTROL_SYNONYMS


def _guide_to_target(gc: str) -> str | None:
    if gc in ("unassigned", "", "nan") or _is_control_synonym(gc):
        return "control"
    if gc.lower().startswith("nontarget") or gc.lower().startswith("non-target"):
        return "control"
    if "|" in gc:
        return None  # dual/multi-guide construct, this single-guide rule doesn't apply
    if not SPACER.search(gc):
        return None
    return SPACER.sub("", gc).replace("_", "-")


def _read_col(grp: h5py.Group, key: str) -> np.ndarray:
    col = grp[key]
    if isinstance(col, h5py.Group):
        codes = col["codes"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in col["categories"][:]])
        out = np.full(codes.shape, "", dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        return out
    raw = col[:]
    if raw.dtype.kind == "S":
        return np.array([v.decode() for v in raw])
    return raw.astype(str)


@dataclass
class ValidationResult:
    dataset: str
    n_labels: int = 0
    n_hyphenated: int = 0
    guide_col: str | None = None
    guide_check: str = "no guide column"  # clean | FLAGGED | inconclusive | no guide column
    merges: dict = field(default_factory=dict)
    zero_hyphen_flag: bool = False
    prefix_suspects: list = field(default_factory=list)
    n_labels_absent_from_var: int = 0
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.guide_check == "FLAGGED"

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.zero_hyphen_flag:
            out.append(
                f"{self.n_labels} labels, 0 hyphenated -- likely truncated "
                f"(every real genome-wide panel has some hyphenated symbols)"
            )
        if self.prefix_suspects:
            out.append(
                f"{len(self.prefix_suspects)} labels absent from var but var "
                f"contains '<label>-*' (unconfirmed -- real false positives "
                f"exist, e.g. GAS5 vs GAS5-AS1): {self.prefix_suspects[:5]}"
            )
        return out


def validate_dataset(path: Path) -> ValidationResult:
    name = path.stem
    result = ValidationResult(dataset=name)
    try:
        with h5py.File(path, "r") as f:
            obs, var = f["obs"], f["var"]
            obs_keys = list(obs.keys())
            var_keys = list(var.keys())

            if "perturbation" not in obs_keys:
                result.error = "no obs['perturbation'] column"
                return result

            perts = _read_col(obs, "perturbation")
            uniq_perts = sorted(set(perts))
            result.n_labels = len(uniq_perts)
            result.n_hyphenated = sum(1 for p in uniq_perts if "-" in p and not _is_control_synonym(p))
            result.zero_hyphen_flag = (
                result.n_labels > ZERO_HYPHEN_MIN_LABELS and result.n_hyphenated == 0
            )

            if "_index" in var_keys:
                var_genes = set(
                    v.decode() if isinstance(v, bytes) else v for v in var["_index"][:]
                )
            elif "gene_name" in var_keys:
                var_genes = set(_read_col(var, "gene_name"))
            else:
                var_genes = set()

            result.n_labels_absent_from_var = sum(
                1 for p in uniq_perts if p not in var_genes and not _is_control_synonym(p)
            )
            result.prefix_suspects = [
                p for p in uniq_perts
                if p not in var_genes and not _is_control_synonym(p)
                and any(g.startswith(p + "-") for g in var_genes)
            ]

            guide_key = next((k for k in GUIDE_COL_CANDIDATES if k in obs_keys), None)
            result.guide_col = guide_key
            if guide_key is None:
                return result

            guides = _read_col(obs, guide_key)
            label_to_targets: dict[str, set[str]] = defaultdict(set)
            n_parsed = 0
            for lbl, gc in zip(perts, guides):
                tgt = _guide_to_target(gc)
                if tgt is not None:
                    n_parsed += 1
                    label_to_targets[lbl].add(tgt)

            if len(perts) and n_parsed / len(perts) < 0.5:
                result.guide_check = "inconclusive (majority unparsed -- different guide-naming convention)"
                return result

            def _is_real_merge(lbl: str, tgts: set[str]) -> bool:
                if _is_control_synonym(lbl) and all(_is_control_synonym(t) for t in tgts):
                    return False
                if len(tgts) > 1:
                    return True
                return next(iter(tgts)) != lbl and not _is_control_synonym(lbl)

            merges = {lbl: sorted(tgts) for lbl, tgts in label_to_targets.items()
                      if _is_real_merge(lbl, tgts)}
            result.merges = merges
            result.guide_check = "FLAGGED" if merges else "clean"
    except Exception as e:  # noqa: BLE001
        result.error = f"error: {e}"
    return result
