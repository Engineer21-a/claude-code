from video_censor.config import (
    AppConfig,
    CensorConfig,
    DetectorConfig,
    IOConfig,
    TrackerConfig,
)
from video_censor.models import CensorMethod


class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.person_model == "yolo11n.pt"
        assert "license-plate" in cfg.license_plate_model or "license_plate" in cfg.license_plate_model
        assert cfg.confidence_threshold == 0.4
        assert cfg.device == "cpu"

    def test_custom_values(self):
        cfg = DetectorConfig(device="cuda", confidence_threshold=0.7)
        assert cfg.device == "cuda"
        assert cfg.confidence_threshold == 0.7


class TestTrackerConfig:
    def test_defaults(self):
        cfg = TrackerConfig()
        assert 0 < cfg.track_thresh < 1
        assert cfg.track_buffer > 0
        assert 0 < cfg.match_thresh < 1
        assert cfg.frame_rate > 0


class TestCensorConfig:
    def test_default_method_is_gaussian(self):
        cfg = CensorConfig()
        assert cfg.method == CensorMethod.GAUSSIAN_BLUR

    def test_default_kernel_size_is_odd(self):
        cfg = CensorConfig()
        assert cfg.blur_kernel_size % 2 == 1

    def test_custom_method(self):
        cfg = CensorConfig(method=CensorMethod.PIXELATE)
        assert cfg.method == CensorMethod.PIXELATE


class TestIOConfig:
    def test_defaults_are_empty_strings(self):
        cfg = IOConfig()
        assert cfg.input_path == ""
        assert cfg.output_path == ""

    def test_default_batch_size_positive(self):
        cfg = IOConfig()
        assert cfg.batch_size > 0


class TestAppConfig:
    def test_defaults_censor_all(self):
        cfg = AppConfig()
        assert cfg.censor_persons is True
        assert cfg.censor_license_plates is True
        assert cfg.censor_logos is True

    def test_nested_configs_are_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.detector, DetectorConfig)
        assert isinstance(cfg.tracker, TrackerConfig)
        assert isinstance(cfg.censor, CensorConfig)
        assert isinstance(cfg.io, IOConfig)

    def test_interactive_selections_default_empty(self):
        cfg = AppConfig()
        assert cfg.interactive_selections == []

    def test_show_progress_default_true(self):
        cfg = AppConfig()
        assert cfg.show_progress is True

    def test_custom_io_config(self):
        io = IOConfig(input_path="/tmp/in.mp4", output_path="/tmp/out.mp4")
        cfg = AppConfig(io=io)
        assert cfg.io.input_path == "/tmp/in.mp4"
