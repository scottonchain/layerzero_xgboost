# Code review copies (temporary)

Plain-text copies of every notebook, generated with jupytext (`py:percent` format: `# %%` starts a code
cell, `# %% [markdown]` a markdown cell). They contain the notebooks' code and markdown only, no outputs,
so they can be read and commented on line by line in the pull request.

The `.ipynb` notebooks remain the source of truth; edits go there. Regenerate after any notebook change:

    for nb in [0-9][0-9]_*.ipynb; do jupytext --to py:percent --opt notebook_metadata_filter=-all \
        --opt cell_metadata_filter=-all -o code_review/${nb%.ipynb}.py $nb; done

This folder is for the authors' review and is removed before the final, cited version.
