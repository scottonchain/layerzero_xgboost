# CLAUDE.md

## Current work: leakage revision

The paper built on this repo is being revised because reviewers raised data leakage. The work list is `docs/REVISION_LEAKAGE.md`: read its Status table first. Record every measured number in its Findings log (section E) with the commit it came from.

What changed and why, in one place:
- `sybil_pipeline.py` is the single source for features, funding groups, and splits. Every notebook imports it. Do not reintroduce per-notebook copies of the pipeline.
- The split is a stratified group split on funding groups (A1). The original random split is trained only in `11_split_comparison`, to measure the leakage.
- Model training, evaluation and the blend weight are defined once in `sybil_pipeline.py` (`fit_xgb`, `fit_lgbm`, `fit_lr`, `select_blend_weight`, `evaluate`); notebooks call them.
- Test labels are used only for final reporting: hyperparameters come from `02_hyperparameter_search` (validation only), thresholds and the blend weight from validation.
- Labeled addresses come from a fixed rule (`sp.stream_labeled_anchors`): public lists, then an Etherscan check of every other funder of at least `ETHERSCAN_MIN_FANOUT` interactors (`review_support/build_etherscan_lookups.py`). No hand additions.
- The model uses 62 features: `gini_coefficient` was removed by a pre-specified test (`01`). `FEATS_SUBMITTED` keeps the submitted 63.
- Provision edges at or after `SNAPSHOT_END` are dropped, and tree features are computed in the pipeline from the filtered network (A7). The precomputed file in `data/20241117_tree_features/` is only a regression check.
- XGBoost's thread count is pinned (`sp.N_JOBS`), because `hist` results depend on it. LightGBM runs with `sp.LGBM_REPRO` (`force_col_wise`, `deterministic`); without it, its trees change between runs.

## Confidentiality

This repo and Paven's fork (paven86/layerzero_xgboost) are public. Never commit the manuscript `.tex`, reviewer comments, or the editor's email. If those files are available, they belong in `private/` (gitignored).

## Running

- Data files are Git LFS objects. Run `git lfs pull` before anything else.
- Install with `pip install -r requirements.txt`.
- Order: run the notebooks in numerical order, `00` → `11`. `01` (Gini keep-or-drop) decides the feature set and `02` (search, about 80 minutes on 4 cores) the hyperparameters; both are one-time steps whose results are committed. `03`–`06` are the models (group split), `07`–`09` robustness, `10` the tie-out of every reported number, `11` the comparison with the original random split. The README has the exact commands.
- Results go to `results/<notebook>.json` with the code commit recorded at kernel start. Start runs from a clean working tree, or the commit is recorded as `-dirty` and `10_tie_out` rejects it.
- `10_tie_out` (and `11` for the results it compares) refuses results whose dependencies (`sybil_pipeline.py`, the notebook, `requirements.txt`, `data/`, the search notebook where used, and for `06` the notebooks whose predictions it reads) changed since they were produced (`sp.provenance`). Editing `sybil_pipeline.py` therefore means rerunning everything, including the search.
- Commit the executed notebooks and `results/`; the paper's numbers are read from them.

## Remotes

`origin` is scottonchain/layerzero_xgboost (the parent). Paven's fork, which the paper cites, is `paven` (read-only from here).
