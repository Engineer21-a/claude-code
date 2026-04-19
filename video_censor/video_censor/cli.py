from __future__ import annotations

import argparse
import os
import sys

from .config import AppConfig, CensorConfig, DetectorConfig, IOConfig, TrackerConfig
from .models import BoundingBox, CensorMethod, InteractiveSelection
from .pipeline import CensorPipeline

# Supported input extensions (OpenCV can open others, but these are common video formats)
_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mts", ".m2ts", ".3gp",
}


def _positive_int(value: str) -> int:
    """argparse type that requires a strictly positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {n}.")
    return n


def _positive_odd_int(value: str) -> int:
    """argparse type for kernel sizes: positive odd integer."""
    n = _positive_int(value)
    # Silently enforce odd; matches the bitwise-OR enforcement in effects.py
    return n | 1


def _confidence(value: str) -> float:
    """argparse type for confidence threshold in [0.0, 1.0]."""
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a float.")
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(
            f"Confidence must be between 0.0 and 1.0, got {f}."
        )
    return f


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
        type=_confidence,
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
        type=_positive_odd_int,
        default=51,
        metavar="SIZE",
        help="Gaussian blur kernel size, must be a positive odd integer (default: 51).",
    )
    censor.add_argument(
        "--pixel-size",
        type=_positive_int,
        default=15,
        metavar="SIZE",
        help="Pixelation block size, must be a positive integer (default: 15).",
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
        type=_positive_int,
        default=8,
        metavar="N",
        help="Frames decoded per batch, must be >= 1 (default: 8).",
    )
    perf.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser


def _validate_paths(input_path: str, output_path: str) -> None:
    """Raise argparse.ArgumentTypeError for invalid or unsafe paths."""
    # --- Input ---
    if not os.path.exists(input_path):
        raise argparse.ArgumentTypeError(
            f"Input file not found: {input_path!r}"
        )
    if not os.path.isfile(input_path):
        raise argparse.ArgumentTypeError(
            f"Input path is not a regular file: {input_path!r}"
        )
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        raise argparse.ArgumentTypeError(
            f"Input file extension {ext!r} is not a recognised video format. "
            f"Supported: {sorted(_VIDEO_EXTENSIONS)}"
        )

    # --- Output ---
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    if not os.path.isdir(output_dir):
        raise argparse.ArgumentTypeError(
            f"Output directory does not exist: {output_dir!r}"
        )
    if not os.access(output_dir, os.W_OK):
        raise argparse.ArgumentTypeError(
            f"Output directory is not writable: {output_dir!r}"
        )
    # Prevent overwriting the input file
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise argparse.ArgumentTypeError(
            "Output path must differ from input path."
        )


def parse_selection(raw: str) -> InteractiveSelection:
    """Parse a --select value of the form 'frame,x1,y1,x2,y2'."""
    try:
        parts = [int(p.strip()) for p in raw.split(",")]
        if len(parts) != 5:
            raise ValueError
        frame_index, x1, y1, x2, y2 = parts
        if frame_index < 0:
            raise argparse.ArgumentTypeError(
                f"frame_index must be >= 0, got {frame_index}."
            )
        if x1 >= x2 or y1 >= y2:
            raise argparse.ArgumentTypeError(
                f"Bounding box is degenerate: x1={x1} x2={x2} y1={y1} y2={y2}. "
                "Require x1 < x2 and y1 < y2."
            )
        return InteractiveSelection(frame_index=frame_index, bbox=BoundingBox(x1, y1, x2, y2))
    except argparse.ArgumentTypeError:
        raise
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
    # Clamp to 100 % — codec metadata can report an incorrect frame count
    pct = min(current / total * 100, 100.0)
    bar_len = 40
    filled = min(int(bar_len * current / total), bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:.1f}% ({current}/{total} frames)", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success, non-zero = error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_paths(args.input, args.output)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    cfg = build_config_from_args(args)

    pipeline = CensorPipeline(cfg)
    try:
        pipeline.build()
    except (MemoryError, RuntimeError, OSError) as exc:
        print(f"Error loading models: {exc}", file=sys.stderr)
        return 1

    on_progress = _progress_bar if cfg.show_progress else None

    try:
        stats = pipeline.run(on_progress=on_progress)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"\nI/O error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
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
