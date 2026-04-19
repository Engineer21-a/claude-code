import argparse
import os
import pytest

from video_censor.cli import (
    _confidence,
    _positive_int,
    _positive_odd_int,
    _validate_paths,
    build_config_from_args,
    build_parser,
    main,
    parse_selection,
)
from video_censor.models import CensorMethod


# ---------------------------------------------------------------------------
# Argument type validators
# ---------------------------------------------------------------------------

class TestPositiveInt:
    def test_valid(self):
        assert _positive_int("5") == 5

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            _positive_int("0")

    def test_negative_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            _positive_int("-1")

    def test_non_integer_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
            _positive_int("abc")


class TestPositiveOddInt:
    def test_odd_value_unchanged(self):
        assert _positive_odd_int("51") == 51

    def test_even_value_made_odd(self):
        assert _positive_odd_int("10") == 11

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_odd_int("0")


class TestConfidence:
    def test_valid_midrange(self):
        assert abs(_confidence("0.5") - 0.5) < 0.001

    def test_boundary_zero(self):
        assert _confidence("0.0") == 0.0

    def test_boundary_one(self):
        assert _confidence("1.0") == 1.0

    def test_above_one_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="between 0.0 and 1.0"):
            _confidence("1.1")

    def test_below_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="between 0.0 and 1.0"):
            _confidence("-0.1")

    def test_non_float_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not a float"):
            _confidence("high")


# ---------------------------------------------------------------------------
# _validate_paths
# ---------------------------------------------------------------------------

class TestValidatePaths:
    def test_valid_paths(self, tmp_path):
        inp = tmp_path / "in.mp4"
        inp.write_bytes(b"")
        out = tmp_path / "out.mp4"
        _validate_paths(str(inp), str(out))  # should not raise

    def test_input_not_found_raises(self, tmp_path):
        with pytest.raises(argparse.ArgumentTypeError, match="not found"):
            _validate_paths(str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4"))

    def test_input_is_directory_raises(self, tmp_path):
        with pytest.raises(argparse.ArgumentTypeError, match="not a regular file"):
            _validate_paths(str(tmp_path), str(tmp_path / "out.mp4"))

    def test_unsupported_extension_raises(self, tmp_path):
        inp = tmp_path / "data.csv"
        inp.write_bytes(b"")
        with pytest.raises(argparse.ArgumentTypeError, match="not a recognised video format"):
            _validate_paths(str(inp), str(tmp_path / "out.mp4"))

    def test_output_dir_missing_raises(self, tmp_path):
        inp = tmp_path / "in.mp4"
        inp.write_bytes(b"")
        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            _validate_paths(str(inp), str(tmp_path / "nonexistent_dir" / "out.mp4"))

    def test_input_equals_output_raises(self, tmp_path):
        inp = tmp_path / "video.mp4"
        inp.write_bytes(b"")
        with pytest.raises(argparse.ArgumentTypeError, match="differ from input"):
            _validate_paths(str(inp), str(inp))

    def test_mkv_extension_accepted(self, tmp_path):
        inp = tmp_path / "in.mkv"
        inp.write_bytes(b"")
        out = tmp_path / "out.mp4"
        _validate_paths(str(inp), str(out))  # should not raise


# ---------------------------------------------------------------------------
# parse_selection
# ---------------------------------------------------------------------------

class TestParseSelection:
    def test_valid_selection_all_fields(self):
        sel = parse_selection("5,100,200,300,400")
        assert sel.frame_index == 5
        assert sel.bbox.x1 == 100
        assert sel.bbox.y1 == 200
        assert sel.bbox.x2 == 300
        assert sel.bbox.y2 == 400

    def test_valid_selection_zero_frame(self):
        sel = parse_selection("0,0,0,640,480")
        assert sel.frame_index == 0

    def test_invalid_too_few_parts(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid --select"):
            parse_selection("5,100,200")

    def test_invalid_too_many_parts(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid --select"):
            parse_selection("5,100,200,300,400,500")

    def test_invalid_non_integer(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid --select"):
            parse_selection("5,100,200,abc,400")

    def test_whitespace_around_values(self):
        sel = parse_selection(" 1 , 10 , 20 , 30 , 40 ")
        assert sel.frame_index == 1
        assert sel.bbox.x1 == 10

    def test_negative_frame_index_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="frame_index must be >= 0"):
            parse_selection("-1,0,0,10,10")

    def test_degenerate_box_x_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="degenerate"):
            parse_selection("0,50,0,50,100")  # x1 == x2

    def test_degenerate_box_y_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="degenerate"):
            parse_selection("0,0,50,100,50")  # y1 == y2


# ---------------------------------------------------------------------------
# build_config_from_args
# ---------------------------------------------------------------------------

def _parse(*args: str) -> argparse.Namespace:
    return build_parser().parse_args(["input.mp4", "output.mp4"] + list(args))


class TestBuildConfigFromArgs:
    def test_default_config(self):
        cfg = build_config_from_args(_parse())
        assert cfg.censor_persons is True
        assert cfg.censor_license_plates is True
        assert cfg.censor_logos is True
        assert cfg.censor.method == CensorMethod.GAUSSIAN_BLUR
        assert cfg.show_progress is True
        assert cfg.detector.device == "cpu"
        assert cfg.io.batch_size == 8

    def test_no_persons_flag(self):
        cfg = build_config_from_args(_parse("--no-persons"))
        assert cfg.censor_persons is False
        assert cfg.censor_license_plates is True

    def test_no_plates_flag(self):
        cfg = build_config_from_args(_parse("--no-plates"))
        assert cfg.censor_license_plates is False

    def test_no_logos_flag(self):
        cfg = build_config_from_args(_parse("--no-logos"))
        assert cfg.censor_logos is False

    def test_method_pixelate(self):
        cfg = build_config_from_args(_parse("--method", "pixelate"))
        assert cfg.censor.method == CensorMethod.PIXELATE

    def test_method_solid_black(self):
        cfg = build_config_from_args(_parse("--method", "solid_black"))
        assert cfg.censor.method == CensorMethod.SOLID_BLACK

    def test_confidence_threshold(self):
        cfg = build_config_from_args(_parse("--confidence", "0.7"))
        assert abs(cfg.detector.confidence_threshold - 0.7) < 0.001

    def test_blur_kernel(self):
        cfg = build_config_from_args(_parse("--blur-kernel", "21"))
        assert cfg.censor.blur_kernel_size == 21

    def test_blur_kernel_even_forced_odd(self):
        cfg = build_config_from_args(_parse("--blur-kernel", "20"))
        assert cfg.censor.blur_kernel_size % 2 == 1

    def test_pixel_size(self):
        cfg = build_config_from_args(_parse("--pixel-size", "20"))
        assert cfg.censor.pixelate_block_size == 20

    def test_padding(self):
        cfg = build_config_from_args(_parse("--padding", "10"))
        assert cfg.censor.padding_px == 10

    def test_single_interactive_selection(self):
        cfg = build_config_from_args(_parse("--select", "10,50,60,200,300"))
        assert len(cfg.interactive_selections) == 1
        assert cfg.interactive_selections[0].frame_index == 10

    def test_multiple_interactive_selections(self):
        cfg = build_config_from_args(_parse("--select", "0,0,0,10,10", "--select", "5,20,20,30,30"))
        assert len(cfg.interactive_selections) == 2
        assert cfg.interactive_selections[0].frame_index == 0
        assert cfg.interactive_selections[1].frame_index == 5

    def test_no_selections_gives_empty_list(self):
        cfg = build_config_from_args(_parse())
        assert cfg.interactive_selections == []

    def test_quiet_disables_progress(self):
        cfg = build_config_from_args(_parse("--quiet"))
        assert cfg.show_progress is False

    def test_device_cuda(self):
        cfg = build_config_from_args(_parse("--device", "cuda"))
        assert cfg.detector.device == "cuda"

    def test_device_mps(self):
        cfg = build_config_from_args(_parse("--device", "mps"))
        assert cfg.detector.device == "mps"

    def test_batch_size(self):
        cfg = build_config_from_args(_parse("--batch-size", "16"))
        assert cfg.io.batch_size == 16

    def test_input_output_paths(self):
        cfg = build_config_from_args(_parse())
        assert cfg.io.input_path == "input.mp4"
        assert cfg.io.output_path == "output.mp4"


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_argument_parser(self):
        assert isinstance(build_parser(), argparse.ArgumentParser)

    def test_prog_name(self):
        assert build_parser().prog == "video-censor"

    def test_invalid_method_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--method", "invalid"])

    def test_invalid_device_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--device", "tpu"])

    def test_zero_batch_size_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--batch-size", "0"])

    def test_negative_batch_size_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--batch-size", "-1"])

    def test_confidence_above_one_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--confidence", "1.5"])

    def test_confidence_below_zero_raises_system_exit(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["input.mp4", "output.mp4", "--confidence", "-0.1"])


# ---------------------------------------------------------------------------
# main() integration — _validate_paths is mocked so no real files needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _skip_path_validation(mocker):
    """Skip filesystem path validation in all main() tests."""
    mocker.patch("video_censor.cli._validate_paths")


class TestMain:
    def test_returns_1_when_build_fails(self, mocker):
        mocker.patch(
            "video_censor.cli.CensorPipeline.build",
            side_effect=RuntimeError("model not found"),
        )
        assert main(["input.mp4", "output.mp4"]) == 1

    def test_returns_1_when_file_not_found(self, mocker):
        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            side_effect=FileNotFoundError("no such file"),
        )
        assert main(["input.mp4", "output.mp4"]) == 1

    def test_returns_1_on_value_error(self, mocker):
        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            side_effect=ValueError("bad video dimensions"),
        )
        assert main(["input.mp4", "output.mp4"]) == 1

    def test_returns_1_on_os_error(self, mocker):
        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            side_effect=OSError("disk full"),
        )
        assert main(["input.mp4", "output.mp4"]) == 1

    def test_returns_1_on_unexpected_error(self, mocker):
        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            side_effect=Exception("unexpected"),
        )
        assert main(["input.mp4", "output.mp4"]) == 1

    def test_returns_0_on_success(self, mocker):
        from video_censor.models import ProcessingStats

        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            return_value=ProcessingStats(
                total_frames=10,
                processed_frames=10,
                total_censored_regions=5,
                elapsed_seconds=1.0,
                fps=10.0,
            ),
        )
        assert main(["input.mp4", "output.mp4", "--quiet"]) == 0

    def test_quiet_suppresses_progress_bar(self, mocker, capsys):
        from video_censor.models import ProcessingStats

        mocker.patch("video_censor.cli.CensorPipeline.build")
        mocker.patch(
            "video_censor.cli.CensorPipeline.run",
            return_value=ProcessingStats(processed_frames=5, fps=5.0),
        )
        main(["input.mp4", "output.mp4", "--quiet"])
        captured = capsys.readouterr()
        assert "Done." in captured.out
