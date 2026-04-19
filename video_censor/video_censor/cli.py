from __future__ import annotations

import argparse
import sys

from .config import AppConfig, CensorConfig, DetectorConfig, IOConfig, TrackerConfig
from .models import BoundingBox, CensorMethod, InteractiveSelection
from .pipeline import CensorPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-censor",
        description=(
            "Automated video censoring for dashcam footage.\n"
            "Censors persons, license plates, logos, and user-selected objects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Censor everything with gaussian blur (default):
  video-censor input.mp4 output.mp4

  # Censor only license plates with pixelation:
  video-censor input.mp4 output.mp4 --no-persons --no-logos --method pixelate

  # Interactively censor an object starting at frame 5 (box x1=100,y1=200,x2=300,y2=400):
  video-censor input.mp4 output.mp4 --select 5,100,200,300,400

  # Multiple interactive selections:
  video-censor input.mp4 output.mp4 --select 0,50,60,150,200 --select 10,300,400,500,600

  # GPU inference:
  video-censor input.mp4 output.mp4 --device cuda
""",
    )

    # Positional arguments
    parser.add_argument("input", help="Path to input dashcam video file.")
    parser.add_argument("output", help="Path for the censored output video file.")

    # Detection toggles
    det = parser.add_argument_group("detection")
    det.add_argument("--no-persons", action="store_true", help="Disable person detection.")
    det.add_argument("--no-plates", action="store_true", help="Disable license plate detection.")
    det.add_argument("--no-logos", action="store_true", help="Disable logo/trademark detection.")
    det.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        metavar="THRESHOLD",
        help="Detection confidence threshold 0.0–1.0 (default: 0.4).",
    )

    # Censoring style
    censor = parser.add_argument_group("censoring")
    censor.add_argument(
        "--method",
        choices=["gaussian_blur", "pixelate", "solid_black"],
        default="gaussian_blur",
        help="Censoring effect (default: gaussian_blur).",
    )
    censor.add_argument(
        "--blur-kernel",
        type=int,
        default=51,
        metavar="SIZE",
        help="Gaussian blur kernel size, must be odd (default: 51).",
    )
    censor.add_argument(
        "--pixel-size",
        type=int,
        default=15,
        metavar="SIZE",
        help="Pixelation block size (default: 15).",
    )
    censor.add_argument(
        "--padding",
        type=int,
        default=5,
        metavar="PX",
        help="Bounding-box padding in pixels (default: 5).",
    )

    # Interactive selection
    interactive = parser.add_argument_group("interactive selection")
    interactive.add_argument(
        "--select",
        action="append",
        metavar="FRAME,X1,Y1,X2,Y2",
        help=(
            "Track and censor a user-selected object. "
            "Format: frame_index,x1,y1,x2,y2. Repeatable."
        ),
    )

    # Hardware
    hw = parser.add_argument_group("hardware")
    hw.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device (default: cpu).",
    )

    # Performance
    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Frames decoded per batch (default: 8).",
    )
    perf.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser


def parse_selection(raw: str) -> InteractiveSelection:
    """Parse a --select value of the form 'frame,x1,y1,x2,y2'."""
    try:
        parts = [int(p.strip()) for p in raw.split(",")]
        if len(parts) != 5:
            raise ValueError
        frame_index, x1, y1, x2, y2 = parts
        return InteractiveSelection(frame_index=frame_index, bbox=BoundingBox(x1, y1, x2, y2))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --select value: {raw!r}. Expected format: frame_index,x1,y1,x2,y2"
        )


def build_config_from_args(args: argparse.Namespace) -> AppConfig:
    """Translate parsed CLI args into an AppConfig."""
    selections: list[InteractiveSelection] = []
    if args.select:
        for raw in args.select:
            selections.append(parse_selection(raw))

    return AppConfig(
        detector=DetectorConfig(
            confidence_threshold=args.confidence,
            device=args.device,
        ),
        tracker=TrackerConfig(),
        censor=CensorConfig(
            method=CensorMethod(args.method),
            blur_kernel_size=args.blur_kernel,
            pixelate_block_size=args.pixel_size,
            padding_px=args.padding,
        ),
        io=IOConfig(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
        ),
        censor_persons=not args.no_persons,
        censor_license_plates=not args.no_plates,
        censor_logos=not args.no_logos,
        interactive_selections=selections,
        show_progress=not args.quiet,
    )


def _progress_bar(current: int, total: int) -> None:
    if total <= 0:
        return
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:.1f}% ({current}/{total} frames)", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success, non-zero = error)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = build_config_from_args(args)

    pipeline = CensorPipeline(cfg)
    try:
        pipeline.build()
    except Exception as exc:
        print(f"Error loading models: {exc}", file=sys.stderr)
        return 1

    on_progress = _progress_bar if cfg.show_progress else None

    try:
        stats = pipeline.run(on_progress=on_progress)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1

    if cfg.show_progress:
        print()

    print(
        f"Done. {stats.processed_frames} frames processed in "
        f"{stats.elapsed_seconds:.1f}s ({stats.fps:.1f} fps). "
        f"{stats.total_censored_regions} regions censored."
    )
    print(f"Output written to: {cfg.io.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
