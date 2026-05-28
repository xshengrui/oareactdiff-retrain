#!/usr/bin/env python
"""Plot validation total loss from checkpoint filenames.

Expected checkpoint filename example:
    ddpm-epoch=000-val-totloss=599.28.ckpt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_CKPT_DIR = Path(
    "/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/"
    "OAReactDiff/oa_reactdiff/trainer/checkpoint/OAReactDiff/leftnet-0-3ef4ea48d1b5"
)

CKPT_RE = re.compile(
    r"epoch=(?P<epoch>\d+)-val-totloss=(?P<loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_checkpoint_name(path: Path) -> tuple[int, float] | None:
    match = CKPT_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("epoch")), float(match.group("loss"))


def collect_losses(ckpt_dir: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for ckpt_path in ckpt_dir.glob("*.ckpt"):
        parsed = parse_checkpoint_name(ckpt_path)
        if parsed is not None:
            points.append(parsed)
    return sorted(points, key=lambda item: item[0])


def filter_loss_values(
    points: list[tuple[int, float]], max_loss: float | None
) -> list[tuple[int, float]]:
    if max_loss is None:
        return points
    return [(epoch, loss) for epoch, loss in points if loss <= max_loss]


def parse_max_loss(value: str) -> float | None:
    if value.lower() in {"none", "no", "false", "off"}:
        return None
    return float(value)


def write_csv(points: list[tuple[int, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "val_totloss"])
        writer.writerows(points)


def plot_losses(points: list[tuple[int, float]], output_png: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    epochs = [epoch for epoch, _ in points]
    losses = [loss for _, loss in points]

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5), dpi=dpi)
    plt.plot(epochs, losses, linewidth=1.4, color="#2563eb")
    plt.xlabel("Epoch")
    plt.ylabel("Validation total loss")
    plt.title("Validation Total Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot val-totloss curve from OAReactDiff checkpoint filenames."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help=f"Directory containing .ckpt files. Default: {DEFAULT_CKPT_DIR}",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output PNG path. Default: <ckpt-dir>/val_totloss_curve.png",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <ckpt-dir>/val_totloss.csv",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--max-loss",
        type=parse_max_loss,
        default=10000.0,
        help="Drop points with val-totloss larger than this before plotting. Use 'none' to disable.",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir.expanduser()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    raw_points = collect_losses(ckpt_dir)
    if not raw_points:
        raise RuntimeError(
            "No checkpoint filenames matched pattern like "
            "'ddpm-epoch=000-val-totloss=599.28.ckpt'."
        )

    points = filter_loss_values(raw_points, args.max_loss)
    if not points:
        raise RuntimeError(
            f"All {len(raw_points)} matched checkpoint points were removed by "
            f"--max-loss {args.max_loss}."
        )

    output_png = args.output_png or ckpt_dir / "val_totloss_curve.png"
    output_csv = args.output_csv or ckpt_dir / "val_totloss.csv"

    write_csv(points, output_csv)
    plot_losses(points, output_png, args.dpi)

    best_epoch, best_loss = min(points, key=lambda item: item[1])
    print(f"Parsed {len(raw_points)} checkpoint files from: {ckpt_dir}")
    if args.max_loss is None:
        print(f"Plotted {len(points)} points without outlier filtering")
    else:
        print(
            f"Plotted {len(points)} points after dropping "
            f"{len(raw_points) - len(points)} points with val-totloss > {args.max_loss:g}"
        )
    print(f"Saved plot: {output_png}")
    print(f"Saved CSV: {output_csv}")
    print(f"Best val-totloss: {best_loss:.6g} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
"""


/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/OAReactDiff/oa_reactdiff/trainer/checkpoint/OAReactDiff/leftnet-0-3ef4ea48d1b5
"""