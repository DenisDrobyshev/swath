**English** · [Русский](README.ru.md)

# swath

[![ci](https://github.com/DenisDrobyshev/swath/actions/workflows/ci.yml/badge.svg)](https://github.com/DenisDrobyshev/swath/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Semantic segmentation of aerial and satellite imagery. From a raw dataset to a page
you can drop a GeoTIFF onto, and a mask that knows where it is on the Earth.

*Swath* is the strip of ground under a sensor on one pass. The package handles what
happens to it afterwards: tiling, training, inference over rasters that do not fit
in video memory, and output a GIS can open.

```
swath prepare loveda --raw data/raw --out data/loveda
swath train --data data/loveda --output runs/landcover
swath evaluate --checkpoint runs/landcover/best.pt --data data/loveda
swath serve --checkpoint runs/landcover/best.pt
```

---

## Contents

| | |
|---|---|
| **Model** | A residual U-Net in one readable file of plain PyTorch. No pretrained backbone, no segmentation library, any number of bands. |
| **Training** | AdamW, cosine schedule with warm-up, mixed precision, class-weighted cross-entropy plus soft Dice, a confusion matrix over the whole validation split, checkpoint selection on mean IoU, resume. |
| **Inference** | Sliding window with raised-cosine blending: tile boundaries do not print onto the result. TTA, a confidence surface, a sieve for speckle. |
| **Georeferencing** | GeoTIFF in, GeoTIFF out: same CRS, same transform. Plus WGS84 polygons and coverage in square metres. |
| **Service** | FastAPI and one page: drag an image in, get an overlay, a confidence view, a legend with coverage, and links to the mask, the GeoTIFF and the GeoJSON. |
| **Checkpoints** | Self-describing. Weights, architecture arguments, class names and palette travel together, so loading one needs nothing but the file. |

## Decisions

**The encoder is not pretrained, and that is a choice.** An ImageNet ResNet would
score a little higher on RGB and would fix the input at three bands. Remote sensing
is not RGB: Sentinel-2 has thirteen bands, and the near-infrared one separates a
healthy field from a bare one. A model taking `in_channels` as an argument is worth
more here than a couple of IoU points.

**Windows are blended, not stitched.** A pixel at the edge of a tile is classified
without half its context, which means worse. Cut a raster up, take the argmax of
each piece and paste it back, and those errors line up into a grid across the map.
An overlap weighted to fall to zero at the edge gives the pixel to the window that
saw the most around it. The effect is measured in `tests/test_predict.py`: on a
model built to be wrong within eight pixels of its border, naive tiling ruins 44% of
the interior, blending 0.3%.

**Metrics come from one confusion matrix over the whole split.** Averaging IoU per
batch is the common shortcut, and it silently scores classes that were not in the
batch. On LoveDA that is not cosmetic: roads are 1.9% of the pixels and agriculture
34%, so the rare classes are missing batch after batch.

**Validation during training runs on crops, the published numbers do not.** Crops
are cheap enough to run every other epoch, and that is all checkpoint selection
needs. A crop is not what a user submits, and it never exercises the sliding window.
`swath evaluate` scores whole tiles through the same predictor the service uses.

## Results

Trained from scratch on the LoveDA training split. Scored on all 1669 validation
tiles at full 1024 px resolution, through the same predictor that sits in the
service.

| class | IoU | F1 | IoU rural | IoU urban |
|---|---|---|---|---|
| background | 0.455 | 0.626 | 0.500 | 0.345 |
| building | 0.507 | 0.673 | 0.415 | 0.564 |
| road | 0.490 | 0.658 | 0.360 | 0.547 |
| water | 0.476 | 0.645 | 0.406 | 0.595 |
| barren | 0.334 | 0.501 | 0.254 | 0.411 |
| forest | 0.374 | 0.545 | 0.252 | 0.471 |
| agriculture | 0.426 | 0.598 | 0.464 | 0.364 |
| **mean** | **0.438** | **0.607** | **0.379** | **0.471** |

![example predictions](docs/examples.png)

![training curves](docs/training.png)

Overall accuracy 0.615, mean IoU **0.438** over 16.8M parameters. Thirty epochs took
2.1 hours on one RTX 4060 Laptop with 8 GB.

Three things worth weighing:

- **The encoder is not pretrained.** Published LoveDA baselines take an ImageNet
  backbone and reach 0.47–0.50 mIoU. Starting from random weights costs accuracy and
  buys a model that takes any number of bands.
- **The validation split both selected the checkpoint and scored it.** LoveDA's test
  split has no public labels: this is what the benchmark allows, and the published
  baselines use the same protocol. The number is mildly optimistic all the same.
- **The domains diverge.** Rural gives 0.379, urban 0.471. Read the gap, not the
  average.

The configuration, the epoch-by-epoch history and the report are in
[`docs/reference-run/`](docs/reference-run): the numbers can be checked rather than
taken on trust.

## Install

```bash
pip install -e ".[geo,service]"
```

There is one heavy requirement: `torch`. The `geo` extra adds rasterio, which is
what the georeferencing rests on. Without it the package trains, predicts and serves
PNG masks, and the GeoTIFF and GeoJSON outputs raise a clear error. `service` adds
FastAPI and uvicorn, `dev` adds pytest and ruff.

For CUDA the matching torch build goes in first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Data

The default corpus is [LoveDA](https://doi.org/10.5281/zenodo.5706578): 0.3 m aerial
imagery over three Chinese cities, 1024×1024 tiles, seven land-cover classes, split
into a rural and an urban domain. No registration, CC BY 4.0.

```bash
mkdir -p data/raw
curl -L -o data/raw/Train.zip https://zenodo.org/records/5706578/files/Train.zip
curl -L -o data/raw/Val.zip   https://zenodo.org/records/5706578/files/Val.zip
swath prepare loveda --raw data/raw --out data/loveda
```

That unpacks 2522 training and 1669 validation tiles. The raw masks use `0` for
no-data and `1…7` for the classes. `swath.data.loveda` maps them onto `0…6` and
sends no-data to the ignore label, which is excluded from both the loss and the
metrics.

The classes are badly balanced, which is what the weighted loss is for. Measured
over 400 training tiles:

| agriculture | forest | background | water | barren | building | road |
|---|---|---|---|---|---|---|
| 34.4% | 25.0% | 24.7% | 7.7% | 3.5% | 2.8% | 1.9% |

Adding another corpus takes one function returning a list of `Sample(image, mask)`
pairs, and a registration:

```python
from swath.data import Corpus, register

register(
    Corpus(
        name="inria",
        title="Inria aerial image labeling",
        default_task="buildings",
        splits=("train", "test"),
        discover=find_inria_tiles,  # (root, split) -> [Sample, ...]
    )
)
```

Every command then takes it: `swath train --dataset inria`,
`swath evaluate --dataset inria`. `swath info` lists what is registered.

## Train

```bash
swath train \
  --data data/loveda --output runs/landcover \
  --epochs 30 --batch-size 6 --crop-size 512 \
  --base-channels 32 --depth 4 --learning-rate 6e-4
```

A run writes `best.pt`, `last.pt`, `history.csv`, `metrics.txt` and the exact
`config.json` it used.

On an 8 GB card this configuration peaks at 5.7 GiB. `--base-channels 48` needs a
smaller batch or a 384 px crop. Going over the limit is worse than it sounds:
Windows starts spilling into system memory and the same step takes forty times
longer.

## Evaluate

```bash
swath evaluate --checkpoint runs/landcover/best.pt --data data/loveda --split Val
swath evaluate --checkpoint runs/landcover/best.pt --domain Rural   # per domain
```

Whole 1024 px tiles through the sliding window, all into one confusion matrix. It
prints a per-class table and writes the same numbers as JSON.

LoveDA keeps the rural and urban domains apart. Scoring each separately shows the
failure land-cover models are prone to: learning one kind of landscape and
collapsing on the other. Averaging hides it.

## Predict

```bash
swath predict --checkpoint runs/landcover/best.pt \
              --input scene.tif --output predictions \
              --overlay --geojson --tta
```

Writes `scene_mask.png`, and for a georeferenced input `scene_mask.tif` in the same
CRS. `--overlay` adds a blended preview, `--geojson` vectorises the mask into WGS84
polygons with areas. `--confidence` saves the winning class probability as a
separate greyscale raster: the quickest way to see where the model is guessing
rather than deciding. `--input` also takes a directory.

`--sieve N` removes predicted regions smaller than N pixels. Any per-pixel
classifier leaves speckle, and sieving is the standard clean-up before vectorising.
It is off by default on purpose: a sieve large enough to tidy a land-cover map will
also erase a single shed.

## Serve

```bash
swath serve --checkpoint runs/landcover/best.pt --port 8000
```

Open `http://127.0.0.1:8000`. Drop an image on the page and pick a model. The
overlay comes back with a legend showing what fraction of the scene each class
covers, and in square metres when the upload was georeferenced. `--checkpoint` takes
a directory or can be repeated, and the page then offers a model picker.

A fourth tab shows the confidence surface. On land cover the class boundaries and
the roofs in shadow are what stand out there. The overlay is composed in the browser
from the mask and the input, so the opacity slider is instant rather than another
model run. What goes to the page is downscaled to 2048 px on the long side: a
64-megapixel mask as base64 weighs tens of megabytes that no screen will show. The
download links serve full resolution.

| Method | Path | |
|---|---|---|
| `GET` | `/api/health` | version, device, model count, whether rasterio is available |
| `GET` | `/api/models` | loaded checkpoints with their classes, palettes and metrics |
| `POST` | `/api/segment` | multipart upload; returns the mask, confidence, rendered input, coverage and links |
| `GET` | `/api/result/{id}/mask.png` | the mask alone |
| `GET` | `/api/result/{id}/mask.tif` | georeferenced mask, when the input carried a CRS |
| `GET` | `/api/result/{id}/mask.geojson` | vectorised polygons in EPSG:4326 |

### Docker

```bash
docker build -t swath .
docker run -p 8000:8000 -v "$PWD/runs:/models" swath
```

A CPU image, serving the checkpoints under `/models`. The path comes from
`SWATH_CHECKPOINTS`, which is what `swath serve` falls back on when `--checkpoint`
is absent, so the container needs no arguments.

## Tasks

A *task* holds everything that separates one segmentation problem from another:
class names, palette, band count, normalisation. The model, the training loop and
the service do not change with it. `swath info` lists the registered ones:
`landcover`, `buildings`, `roads`, `greenery`, `flood`.

```python
from swath.tasks import TASKS, Task

TASKS.register(
    Task(
        name="solar",
        title="Rooftop photovoltaics",
        description="Solar panels on roofs.",
        classes=("background", "panel"),
        palette=((255, 255, 255), (255, 140, 0)),
    )
)
```

Point a dataset at it and train. The rest of the code does not need to know.

## Tests

```bash
pytest -q                      # unit and integration
ruff check src tests scripts   # lint
python scripts/smoke_test.py   # trains, predicts and serves end to end
```

The smoke test runs on synthetic tiles, so CI exercises the whole path on every push
without downloading anything.

## Licence

Code: MIT, see [LICENSE](LICENSE).

LoveDA is CC BY 4.0 — Wang, Junjue et al., *LoveDA: A Remote Sensing Land-Cover
Dataset for Domain Adaptive Semantic Segmentation*, NeurIPS 2021 Datasets and
Benchmarks. Weights trained on it inherit the attribution requirement.
