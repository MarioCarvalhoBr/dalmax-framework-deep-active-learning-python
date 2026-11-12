# assets/

`active-learning-framework.png` is the README figure: the pool-based active learning loop, the dataset
sizes of `daninhas_full_fixed`, the query-strategy families of the benchmark, the lab methods, the controls
and the protocol of the clean campaign. Its source is the TikZ file `src/active-learning-framework.tex`,
which is self-contained (no images or other inputs), so the figure can be rebuilt from tracked files only.

Rebuild from the repository root (needs `tectonic` and `pdftoppm` from poppler-utils):

```bash
tectonic -X compile --outdir /tmp assets/src/active-learning-framework.tex
pdftoppm -png -singlefile -scale-to-x 1800 -scale-to-y -1 \
    /tmp/active-learning-framework.pdf assets/active-learning-framework
```

The PNG is exported 1800 px wide (2x the about 900 px README width) on an opaque white background.
Keep the file name: `README.md` links to it. The figure carries no result numbers; when the dataset,
the strategy list or the campaign size change, edit the `.tex` and rebuild.
