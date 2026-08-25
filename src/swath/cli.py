"""Command line interface.

Six verbs cover the whole path: ``prepare`` unpacks and indexes a corpus,
``train`` fits a model, ``evaluate`` scores one at full resolution, ``predict``
runs it over a raster, ``serve`` puts it behind a web page, and ``info`` says
what a checkpoint contains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from swath import __version__
from swath.data.corpora import names as corpus_names
from swath.tasks import TASKS, get_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swath",
        description="Semantic segmentation of aerial and satellite imagery.",
    )
    parser.add_argument("--version", action="version", version=f"swath {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="unpack and index a dataset")
    prepare.add_argument("dataset", choices=corpus_names(), help="which corpus to prepare")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw"), help="downloaded archives")
    prepare.add_argument("--out", type=Path, default=Path("data/loveda"), help="unpack target")
    prepare.set_defaults(handler=_prepare)

    train = subparsers.add_parser("train", help="train a model")
    train.add_argument(
        "--dataset", default="loveda", choices=corpus_names(), help="which corpus to train on"
    )
    train.add_argument("--task", default=None, help="task name; defaults to the corpus task")
    train.add_argument("--data", type=Path, default=Path("data/loveda"), help="prepared corpus")
    train.add_argument(
        "--output", type=Path, default=Path("runs/landcover"), help="where the run is written"
    )
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--crop-size", type=int, default=512)
    train.add_argument("--val-crop-size", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--base-channels", type=int, default=48)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--val-interval", type=int, default=1, help="validate every N epochs")
    train.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--limit-train", type=int, default=0, help="cap training tiles (debug)")
    train.add_argument("--limit-val", type=int, default=0, help="cap validation tiles (debug)")
    train.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue from a checkpoint, restoring the optimiser and epoch counter",
    )
    train.add_argument("--notes", default="", help="free text stored in the checkpoint")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="score a checkpoint on a labelled split, at full resolution"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="model to score")
    evaluate.add_argument("--dataset", default="loveda", choices=corpus_names())
    evaluate.add_argument("--data", type=Path, default=Path("data/loveda"), help="prepared corpus")
    evaluate.add_argument("--split", default="Val", help="which split to score")
    evaluate.add_argument("--domain", default="both", help="restrict to one domain")
    evaluate.add_argument("--tile", type=int, default=512)
    evaluate.add_argument("--overlap", type=int, default=128)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    evaluate.add_argument("--tta", action="store_true", help="average over flips and rotations")
    evaluate.add_argument(
        "--sieve", type=int, default=0, help="drop predicted regions smaller than N pixels"
    )
    evaluate.add_argument("--limit", type=int, default=0, help="score only the first N tiles")
    evaluate.add_argument("--report", type=Path, default=None, help="write the metrics as JSON")
    evaluate.set_defaults(handler=_evaluate)

    predict = subparsers.add_parser("predict", help="segment a raster")
    predict.add_argument("--checkpoint", type=Path, required=True, help="trained model")
    predict.add_argument("--input", type=Path, required=True, help="image file or a directory")
    predict.add_argument(
        "--output", type=Path, default=Path("predictions"), help="where to write the results"
    )
    predict.add_argument("--tile", type=int, default=512, help="sliding window size")
    predict.add_argument("--overlap", type=int, default=128, help="how far windows overlap")
    predict.add_argument("--batch-size", type=int, default=4, help="windows per forward pass")
    predict.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="where to run"
    )
    predict.add_argument("--tta", action="store_true", help="average over flips and rotations")
    predict.add_argument("--overlay", action="store_true", help="also write a blended preview")
    predict.add_argument("--geojson", action="store_true", help="also vectorise the mask")
    predict.add_argument(
        "--sieve", type=int, default=0, help="drop predicted regions smaller than N pixels"
    )
    predict.add_argument(
        "--confidence",
        action="store_true",
        help="also write the winning probability per pixel, as a greyscale raster",
    )
    predict.set_defaults(handler=_predict)

    serve = subparsers.add_parser("serve", help="run the web service")
    serve.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="checkpoint or directory to expose; repeat for several models. "
        "Defaults to the SWATH_CHECKPOINTS environment variable.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    serve.set_defaults(handler=_serve)

    info = subparsers.add_parser(
        "info", help="describe a checkpoint, or list the tasks and datasets"
    )
    info.add_argument(
        "--checkpoint", type=Path, default=None, help="describe this checkpoint instead"
    )
    info.set_defaults(handler=_info)

    return parser


def _prepare(args: argparse.Namespace) -> int:
    from swath.data.corpora import get_corpus

    corpus = get_corpus(args.dataset)
    if corpus.prepare is None:
        print(f"{corpus.name} needs no preparation step", file=sys.stderr)
        return 1

    summary = corpus.prepare(args.raw, args.out)
    for split, count in summary.items():
        print(f"{split:6s} {count:5d} tiles")
    print(f"prepared under {args.out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from swath.data.corpora import get_corpus
    from swath.data.dataset import SegmentationDataset
    from swath.data.transforms import build_eval_transform, build_train_transform
    from swath.train import TrainConfig, Trainer

    corpus = get_corpus(args.dataset)
    task = get_task(args.task or corpus.default_task)
    train_samples = corpus.samples(args.data, corpus.train_split)
    val_samples = corpus.samples(args.data, corpus.val_split)
    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]

    mapping = corpus.labels()
    train_dataset = SegmentationDataset(
        train_samples, task, transform=build_train_transform(args.crop_size), label_map=mapping
    )
    val_dataset = SegmentationDataset(
        val_samples,
        task,
        transform=build_eval_transform(args.val_crop_size),
        label_map=mapping,
    )

    config = TrainConfig(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        val_crop_size=args.val_crop_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        depth=args.depth,
        num_workers=args.num_workers,
        val_interval=args.val_interval,
        device=args.device,
        seed=args.seed,
        amp=not args.no_amp,
        resume=args.resume,
        notes=args.notes,
        extra={"dataset": corpus.name, "data_dir": str(args.data)},
    )
    result = Trainer(task, train_dataset, val_dataset, config).fit()
    print(f"best mean IoU {result['best_mean_iou']:.4f}")
    print(f"checkpoints in {args.output}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from swath.checkpoints import load_checkpoint
    from swath.data.corpora import get_corpus
    from swath.evaluate import evaluate, write_report

    model, meta = load_checkpoint(args.checkpoint, map_location="cpu")
    corpus = get_corpus(args.dataset)
    samples = corpus.samples(args.data, args.split, args.domain)
    if args.limit:
        samples = samples[: args.limit]

    report = evaluate(
        model,
        samples,
        meta.task,
        label_map=corpus.labels(),
        tile=args.tile,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        tta=args.tta,
        sieve=args.sieve,
    )
    print(report.summary())

    destination = args.report or args.checkpoint.parent / f"evaluation_{args.split.lower()}.json"
    write_report(destination, report)
    print(f"report written to {destination}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from swath.checkpoints import load_checkpoint
    from swath.geo import class_areas, mask_to_geojson, write_mask_geotiff
    from swath.imagery import overlay, save_png
    from swath.predict import predict_file

    model, meta = load_checkpoint(args.checkpoint, map_location="cpu")
    task = meta.task

    inputs = _collect_inputs(args.input)
    if not inputs:
        print(f"no readable rasters at {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        try:
            result = predict_file(
                model,
                path,
                task,
                return_confidence=args.confidence,
                sieve=args.sieve,
                tile=args.tile,
                overlap=args.overlap,
                batch_size=args.batch_size,
                device=args.device,
                tta=args.tta,
                progress=True,
            )
        except ValueError as error:
            print(f"{path.name}: {error}", file=sys.stderr)
            return 1

        confidence = None
        if args.confidence:
            image, mask, confidence, reference = result
        else:
            image, mask, reference = result

        stem = path.stem
        save_png(args.output / f"{stem}_mask.png", mask)
        if confidence is not None:
            # Scaled to bytes: a confidence raster is for looking at and for
            # thresholding, and eight bits is finer than either needs.
            scaled = (confidence * 255).round().clip(0, 255).astype("uint8")
            save_png(args.output / f"{stem}_confidence.png", scaled)
            if reference is not None:
                write_mask_geotiff(args.output / f"{stem}_confidence.tif", scaled, reference, None)
        if args.overlay:
            save_png(args.output / f"{stem}_overlay.png", overlay(image, mask, task.palette))
        if reference is not None:
            write_mask_geotiff(args.output / f"{stem}_mask.tif", mask, reference, task)
        if args.geojson:
            payload = mask_to_geojson(mask, reference, task)
            (args.output / f"{stem}.geojson").write_text(json.dumps(payload), encoding="utf-8")
        areas = class_areas(mask, task, reference)
        top = sorted(areas, key=lambda row: row["share"], reverse=True)[:3]
        shares = ", ".join(f"{row['class']} {row['share']:.1%}" for row in top if row["share"])
        if confidence is not None:
            shares += f" | mean confidence {float(confidence.mean()):.3f}"
        print(f"{path.name}: {shares}")

    print(f"written to {args.output}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from swath.service.app import create_app

    checkpoints = args.checkpoint or _checkpoints_from_environment()
    if not checkpoints:
        print(
            "no checkpoints given; pass --checkpoint or set SWATH_CHECKPOINTS",
            file=sys.stderr,
        )
        return 1
    app = create_app(checkpoints, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _info(args: argparse.Namespace) -> int:
    if args.checkpoint:
        from swath.checkpoints import describe

        print(describe(args.checkpoint))
        return 0

    from swath.data.corpora import all_corpora

    print("tasks")
    for task in sorted(TASKS, key=lambda item: item.name):
        print(f"  {task.name:12s} {task.num_classes} classes  {task.title}")
        print(f"  {'':12s} {task.description}")

    print()
    print("datasets")
    for corpus in all_corpora():
        splits = ", ".join(corpus.splits)
        print(f"  {corpus.name:12s} task {corpus.default_task}  splits {splits}")
        if corpus.source:
            print(f"  {'':12s} {corpus.source}")
    return 0


def _checkpoints_from_environment() -> list[Path]:
    """Read SWATH_CHECKPOINTS, the deployment path for containers.

    A container is configured with environment variables, not command lines, so
    the image can declare where checkpoints will be mounted and the entrypoint
    stays the same whether one model is served or five.
    """
    raw = os.environ.get("SWATH_CHECKPOINTS", "").strip()
    if not raw:
        return []
    return [Path(part) for part in raw.split(os.pathsep) if part.strip()]


def _collect_inputs(path: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
