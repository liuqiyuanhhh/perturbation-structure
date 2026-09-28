"""Step 02 -- write the knowledge-source list PRESAGE reads.

model.pathway_files points at a plain text file, one absolute path per line;
read_and_embed turns each line into one channel of the gene embedding tensor and
prepends two channels computed from the training data (transpose_matrix and
coexpression), so a list of N lines yields N + 2 channels.

Layout:
   1-31   MSigDB 2023.2 and STRING           (config.PATHWAY_EMB_FILES)
  32-40   Periscope x3, DepMap, Funk OPS,
          GenePT x2, BioGPT, ESM2            (config.OTHER_EMB_FILES)
  41-..   one line per external Perturb-seq screen, from step 01

The 40 base sources keep upstream's order.  The list is content-addressed
(sources.<sha8>.txt, with current.<dataset>.txt naming the live one) because
read_and_embed keys its tensor cache on the list's basename and the dataset
name, not on the list's contents.

Run: PP_DATASET=VCC python 02_make_source_list.py
"""

import hashlib
import json

from config import (
    DATASET,
    OTHER_EMB_DIR,
    OTHER_EMB_FILES,
    PATHWAY_EMB_DIR,
    PATHWAY_EMB_FILES,
    PERT_SRC_DIR,
    SOURCE_LIST_DIR,
    SOURCE_LIST_POINTER,
    pert_source_pkl,
)


def resolve(directory, names):
    paths = [directory / n for n in names]
    absent = [p for p in paths if not p.exists()]
    if absent:
        raise SystemExit("missing knowledge sources:\n  " + "\n  ".join(str(p) for p in absent))
    return paths


def main():
    SOURCE_LIST_DIR.mkdir(parents=True, exist_ok=True)

    # The screens come from THIS target's manifest, never from globbing the
    # shared PERT_SRC_DIR, which holds screens this target may not see.
    manifest_path = PERT_SRC_DIR / f"manifest.{DATASET}.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing -- run 01_build_pert_sources.py first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["target"] != DATASET:
        raise SystemExit(f"{manifest_path} is for {manifest['target']!r}, not {DATASET!r}")
    pert_sources = [pert_source_pkl(e["dataset"]) for e in manifest["sources"]]
    if not pert_sources:
        raise SystemExit(f"{manifest_path} lists no sources")
    absent = [p for p in pert_sources if not p.exists()]
    if absent:
        raise SystemExit("manifest names missing files:\n  " + "\n  ".join(str(p) for p in absent))
    if DATASET in {e["dataset"] for e in manifest["sources"]}:
        raise SystemExit(f"{manifest_path} lists {DATASET} as its own prior")

    pathway = resolve(PATHWAY_EMB_DIR, PATHWAY_EMB_FILES)
    other = resolve(OTHER_EMB_DIR, OTHER_EMB_FILES)
    sources = pathway + other + pert_sources

    body = "\n".join(str(p) for p in sources) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()[:8]
    list_path = SOURCE_LIST_DIR / f"sources.{digest}.txt"
    list_path.write_text(body)
    SOURCE_LIST_POINTER.write_text(list_path.name + "\n")

    print(f"dataset: {DATASET}")
    print(f"\n  {len(pathway):>3} MSigDB / STRING")
    print(f"  {len(other):>3} pretrained-model and other experimental")
    print(f"  {len(pert_sources):>3} cross-dataset Perturb-seq:")
    for p in pert_sources:
        print(f"        {p.name}")
    print(f"  {len(sources):>3} sources total -> {len(sources) + 2} model channels")
    print(f"\nwrote {list_path}")
    print(f"wrote {SOURCE_LIST_POINTER} -> {list_path.name}")


if __name__ == "__main__":
    main()
