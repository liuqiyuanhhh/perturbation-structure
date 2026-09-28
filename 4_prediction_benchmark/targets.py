"""The 12 prediction targets and the other screens each one may learn from.

PRESAGE (presage_pipeline/) takes its Perturb-seq priors from the source pool
and the evaluation (evaluation/) its baselines and scoring window, so both see
the same screens:

    source pool(target) = the 12 screens - the target - its sisters

Sisters would leak the target: the same experiment resampled, or a screen of
the same cell type whose perturbations overlap it heavily.  They are the
sisters of the dataset registry (utils.paths), as effect-dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from utils.paths import DATASET_COMPONENTS, effect_key, sisters

# the 12 screens, as effect-dict keys
SCREENS = tuple(effect_key(name) for name in DATASET_COMPONENTS)


@dataclass(frozen=True)
class Target:
    # effect-dict keys of the sister screens (from the registry, below)
    sisters: tuple = ()
    # GEARS build of GEARS / scGPT-ft ("{fold}" marks one build per fold);
    # None: the two models are not run for this target
    gears_build: str | None = None
    # GEARS build PRESAGE trains on (default: the target name)
    presage_build: str | None = None
    # build whose cv5 split files define the folds (default: presage_build)
    split_build: str | None = None
    # control subtracted from the GEARS / scGPT-ft predictions: the pooled
    # pooled control mean of this build (evaluation/evaluate.py), or,
    # when None, uns/control_profile of the target's moments file
    control_build: str | None = None
    # effect-dict key of the ground truth (default: the target name)
    truth_key: str | None = None


# The per-fold builds of one parent share their control cells, so fold 1's
# control profile serves all five folds.
TARGETS = {
    "VCC": Target(gears_build="VCC"),
    "Pan-GW-hESC": Target(gears_build="Pan-GW-hESC"),
    "Replogle-E-k562": Target(gears_build="Replogle-E-k562"),
    "Replogle-GW-k562": Target(gears_build="Replogle-GW-k562", control_build="Replogle-GW-k562"),
    "Replogle-E-rpe1": Target(gears_build="Replogle-E-rpe1"),
    "Nadig-HEPG2": Target(gears_build="Nadig-HEPG2"),
    "Nadig-JURKAT": Target(gears_build="Nadig-JURKAT"),
    "Feng-ts": Target(gears_build="Feng-ts"),
    "Feng-gw": Target(gears_build="Feng-gw-control-cap", control_build="Feng-gw-control-cap"),
    "Nourreddine-GW-ipsc": Target(gears_build="Nourreddine-GW-ipsc-third-f{fold}",
                                  control_build="Nourreddine-GW-ipsc-third-f1"),
    "Huang-HCT116": Target(gears_build="Huang-HCT116-third-f{fold}",
                           control_build="Huang-HCT116-third-f1"),
    "Huang-HEK293T": Target(gears_build="Huang-HEK293T-third-f{fold}",
                            control_build="Huang-HEK293T-third-f1"),
}
# the sisters of each target in the registry
TARGETS = {
    name: replace(target, sisters=tuple(effect_key(sister) for sister in sisters(name)))
    for name, target in TARGETS.items()
}


def get(name: str) -> Target:
    return TARGETS[name]


def presage_build(name: str) -> str:
    return get(name).presage_build or name


def split_build(name: str) -> str:
    return get(name).split_build or presage_build(name)


def truth_key(name: str) -> str:
    return get(name).truth_key or name


def excluded(name: str) -> tuple:
    """Effect-dict keys of the screens that may not inform `name`, besides its own."""
    return get(name).sisters


def source_pool(effect_keys, name: str) -> list:
    """The other screens `name` may learn from, sorted."""
    drop = {name, truth_key(name), *excluded(name)}
    return sorted(k for k in effect_keys if k in SCREENS and k not in drop)
