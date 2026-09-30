# CLAUDE.md

## Current work: leakage revision

The paper built on this repo is being revised because reviewers raised data leakage. The work list is `docs/REVISION_LEAKAGE.md`. Read it first. Record every measured number in its Findings log (section E) along with the commit hash it came from.

Key findings so far:
- The split is random at the address level (`train_test_split(..., stratify=y)`). Wallets from the same funding cluster share provider and tree feature values and land on both sides of the split. The fix is a stratified group split. The group key is defined in A1.
- `04_cross_ensemble_sybil.ipynb` picks the blend weight on test F1, not validation (A3).
- The hyperparameter search code is not in the repo, and the paper's tuning tables report test F1 (A4).
- Paper vs. code mismatches: IxI and IxE counts are swapped in Table 5. The ensemble F1 in the README (0.741) doesn't match the notebook (0.738). See section C.

## Confidentiality

This repo and Paven's fork (paven86/layerzero_xgboost) are public. Never commit the manuscript `.tex`, reviewer comments, or the editor's email. If those files are available, they belong in `private/` (gitignored).

## Running

- Data files are Git LFS objects. Run `git lfs pull` before anything else.
- Install with `pip install -r requirements.txt`. Model libraries are pinned because their default hyperparameters change between releases.
- `00_data_pipeline.ipynb` writes `output/` (gitignored) in about 40 s. Notebooks 01 to 04 each re-run the pipeline internally and take several minutes each on 4 cores.
- Headless run: `jupyter nbconvert --to notebook --execute <nb> --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=3600`
- When a fix is done, re-execute every affected notebook and commit its saved outputs, because the paper's numbers are read from them.

## Remotes

`origin` is scottonchain/layerzero_xgboost (the parent). Paven's fork, which the paper cites, is `paven` (read-only from here).
