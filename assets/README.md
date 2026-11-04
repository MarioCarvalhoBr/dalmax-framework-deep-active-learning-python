# assets/

`active-learning-framework.png` is the README figure, drawn in TikZ (`active-learning-framework.tex`).
The five patch thumbnails it embeds (`samples/*.jpg`, one 128 x 128 patch per class) are NOT tracked,
because the data-safety rule forbids committing dataset images; copy any five patches from
`DATA/daninhas_full/train/DATASET_*/` into `samples/` with the names used in the `.tex` before rebuilding.
Rebuild from this directory: `tectonic -X compile active-learning-framework.tex` (or `pdflatex`), then
`pdftoppm -r 200 -png -singlefile active-learning-framework.pdf active-learning-framework`.
