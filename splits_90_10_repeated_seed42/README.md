# CAMELYON16 repeated stratified 90/10 splits

These are **five repeated random 90/10 train/validation splits**, **not 5-fold cross-validation**. Validation subjects may appear in more than one repeat. `splits_5fold_<run>.csv` names are used only because the existing loader recognizes this convention.

Point your training CLI at this directory with `--split-dir` and `--split-scheme 5fold --fold <run>`; do not modify the original fivefold split files. Select best epochs by validation AUC only, and use the official test set solely for final evaluation.

See `protocol.json` for seed, data hashes, and exact case membership.
