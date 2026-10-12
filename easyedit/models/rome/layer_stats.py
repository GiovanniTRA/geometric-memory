"""
Stub. The full `layer_stats` (covariance-statistics collection over a text
corpus, used by ROME's optional inverse-second-moment adjustment) is dead
code for every hparams file in this repo: `mom2_adjustment: false` in
hparams/rome_llama3.1-8b.yaml means `compute_u.py` never calls it (see the
`if hparams.mom2_adjustment:` guard there). It's only imported, never
invoked, so a stub avoids vendoring ~2000 lines of EasyEdit (runningstats.py,
tok_dataset.py) and the `datasets`/`tqdm` dependencies they pull in, for a
code path this repo never exercises.
"""


def layer_stats(*args, **kwargs):
    raise NotImplementedError(
        "layer_stats was not vendored into this trimmed repo (see this file's "
        "docstring) -- it is only reachable when hparams.mom2_adjustment is "
        "True, which no hparams file here sets."
    )
