**English** · [Русский](README.ru.md)

# swath

[![ci](https://github.com/DenisDrobyshev/swath/actions/workflows/ci.yml/badge.svg)](https://github.com/DenisDrobyshev/swath/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Semantic segmentation of aerial and satellite imagery — from a raw dataset to a
web page you can drop a GeoTIFF onto and get back a mask that still knows where
it is on the Earth.

*Swath* is the strip of ground a sensor images as it passes overhead. This
package handles what happens to that strip afterwards: tiling it, training a
model on it, running that model over rasters far larger than the GPU, and
handing the result back as something a GIS can open.

```
swath prepare loveda --raw data/raw --out data/loveda
swath train --data data/loveda --output runs/landcover
swath evaluate --checkpoint runs/landcover/best.pt --data data/loveda
swath serve --checkpoint runs/landcover/best.pt
```

---

## What is in the box

| | |
|---|---|
| **Model** | A residual U-Net in one readable file of plain PyTorch. No pretrained backbone, no segmentation library, any number of input bands. |
| **Training** | AdamW, cosine schedule with warm-up, mixed precision, class-weighted cross-entropy plus soft Dice, per-epoch confusion matrix, best-checkpoint selection on mean IoU. |
| **Inference** | Sliding window with raised-cosine blending, so a raster of any size comes out without tile seams. Optional test-time augmentation over flips and quarter turns. |
| **Georeferencing** | A GeoTIFF in, a GeoTIFF out — same CRS, same transform — plus WGS84 polygons and per-class coverage in square metres. |
| **Service** | A FastAPI application with a single-page front end: drag an image in, get an overlay, a legend with coverage, and download links for the mask, the GeoTIFF and the GeoJSON. |
| **Checkpoints** | Self-describing. Weights, architecture arguments, class names and palette travel together, so loading one needs nothing but the file. |

## Why it is built this way

**The model has no pretrained encoder.** An ImageNet-pretrained ResNet would
score a little higher on RGB, and would also fix the input at three bands.
Remote sensing is not RGB — Sentinel-2 has thirteen bands, and the near-infrared
one is what separates a healthy field from a bare one. A model that takes
`in_channels` as an argument is worth more here than a couple of IoU points.

**Overlapping windows are blended, not stitched.** A pixel near the edge of a
tile is classified with half its context missing, so it is classified worse. Cut
a raster into tiles, take the argmax of each, and paste them back, and those
mistakes line up into a grid across the output. Overlapping the windows and
blending their probabilities with a weight that falls to zero at the tile edge
means every pixel is decided mostly by the window that saw the most around it.
The effect is measured in `tests/test_predict.py`: on a model built to be wrong
within eight pixels of its tile border, naive tiling leaves 44% of the interior
wrong, and blending brings that to 0.3%.

**Metrics come from one confusion matrix per epoch.** Averaging per-batch IoU is
the common shortcut, and it silently scores classes that were not in the batch.
On land cover, where roads are a few percent of the pixels and agriculture is
half of them, that difference is not cosmetic.

**Validation during training runs on crops; reported numbers do not.** Crops are
cheap enough to run every other epoch, which is what checkpoint selection needs.
They are also not what a user submits, and they never exercise the sliding
window. `swath evaluate` scores whole tiles through the same predictor the
service uses.

## Install

```bash
pip install -e ".[geo,service]"
```

`torch` is the only heavy requirement. `geo` adds rasterio, which is what makes
the georeferenced paths work — without it the package still trains, predicts and
serves PNG masks, and the GeoTIFF and GeoJSON outputs raise a clear error
instead. `service` adds FastAPI and uvicorn. `dev` adds pytest and ruff.

For CUDA, install the matching torch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Data

The default corpus is [LoveDA](https://doi.org/10.5281/zenodo.5706578): 0.3 m
aerial imagery over three Chinese cities, 1024×1024 tiles, seven land-cover
classes, split into a rural and an urban domain. It needs no registration and is
CC BY 4.0.

```bash
mkdir -p data/raw
curl -L -o data/raw/Train.zip https://zenodo.org/records/5706578/files/Train.zip
curl -L -o data/raw/Val.zip   https://zenodo.org/records/5706578/files/Val.zip
swath prepare loveda --raw data/raw --out data/loveda
```

That unpacks 2522 training and 1669 validation tiles. The raw masks use `0` for
no-data and `1…7` for the classes; `swath.data.loveda` remaps them onto
`0…6` and sends no-data to the ignore label, which is then excluded from both
the loss and the metrics.

Adding another corpus means writing one function that returns a list of
`Sample(image, mask)` pairs and registering it:

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

The run writes `best.pt`, `last.pt`, `history.csv`, `metrics.txt` and the exact
`config.json` it used. On an 8 GB GPU that configuration peaks at 5.7 GiB;
`--base-channels 48` needs a smaller batch or a 384 px crop to stay under the
limit, and going over it is worse than it sounds — Windows will spill into system
memory and the same step takes forty times longer.

## Evaluate

```bash
swath evaluate --checkpoint runs/landcover/best.pt --data data/loveda --split Val
swath evaluate --checkpoint runs/landcover/best.pt --domain Rural   # per domain
```

Full 1024 px tiles through the sliding-window predictor, into one confusion
matrix. Prints a per-class table and writes the same numbers as JSON. Because
LoveDA keeps the rural and the urban domain apart, scoring them separately shows
the failure land-cover models are prone to — learning one kind of landscape and
collapsing on the other — instead of averaging it away.

## Predict

```bash
swath predict --checkpoint runs/landcover/best.pt \
              --input scene.tif --output predictions \
              --overlay --geojson --tta
```

Writes `scene_mask.png`, and, when the input is georeferenced, `scene_mask.tif`
in the same CRS. `--overlay` adds a blended preview, `--geojson` vectorises the
mask into WGS84 polygons with per-feature areas, and `--confidence` writes the
winning probability per pixel as a greyscale raster — the quickest way to see
where a model is guessing rather than deciding. `--input` also accepts a
directory.

## Serve

```bash
swath serve --checkpoint runs/landcover/best.pt --port 8000
```

Open `http://127.0.0.1:8000`. Drop an image on the page, pick a model, and the
overlay comes back with a legend showing what fraction of the scene each class
covers — in square metres when the upload was georeferenced. `--checkpoint`
takes a directory, or can be repeated, and the page offers a model picker.

A fourth view shows the confidence surface — the winning probability per pixel —
which on land cover is where the class boundaries and the shadowed roofs are.
The overlay is composed in the browser from the mask and the input, so the
opacity slider is instant instead of a round trip through the model. What the
page receives is downscaled to 2048 px on the long side — a 64 megapixel mask
base64-encoded into a JSON body is tens of megabytes no screen can show — while
the download links behind each result stay at full resolution.

The API underneath:

| Method | Path | |
|---|---|---|
| `GET` | `/api/health` | version, device, model count, whether rasterio is available |
| `GET` | `/api/models` | loaded checkpoints with their classes, palettes and metrics |
| `POST` | `/api/segment` | multipart upload; returns the mask, a confidence surface, a rendering of the input, coverage and download links |
| `GET` | `/api/result/{id}/mask.png` | the mask on its own |
| `GET` | `/api/result/{id}/mask.tif` | georeferenced mask, when the input carried a CRS |
| `GET` | `/api/result/{id}/mask.geojson` | vectorised polygons in EPSG:4326 |

### Docker

```bash
docker build -t swath .
docker run -p 8000:8000 -v "$PWD/runs:/models" swath
```

CPU image; it serves whatever checkpoints are mounted at `/models`. The path
comes from `SWATH_CHECKPOINTS`, which `swath serve` falls back to when
`--checkpoint` is not given, so the container needs no arguments.

## Tasks

A *task* holds everything that differs between segmentation problems — class
names, palette, band count, normalisation — so the model, the training loop and
the service stay identical across them. `swath info` lists the registered ones:
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

Point a dataset at it and train; nothing else needs to know.

## Tests

```bash
pytest -q          # unit and integration
ruff check .       # lint
python scripts/smoke_test.py   # trains, predicts and serves end to end
```

The smoke test runs on synthetic tiles, so CI exercises the whole path on every
push without downloading a dataset.

## Licence

Code: MIT, see [LICENSE](LICENSE).

LoveDA is CC BY 4.0 — Wang, Junjue et al., *LoveDA: A Remote Sensing Land-Cover
Dataset for Domain Adaptive Semantic Segmentation*, NeurIPS 2021 Datasets and
Benchmarks. Weights trained on it inherit that attribution requirement.
