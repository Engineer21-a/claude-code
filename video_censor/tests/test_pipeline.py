import numpy as np
import pytest

from video_censor.config import AppConfig, CensorConfig, IOConfig
from video_censor.models import (
    BoundingBox,
    CensorMethod,
    Detection,
    DetectionClass,
    FrameDetections,
)
from video_censor.pipeline import CensorPipeline, _merge_batch_detections


# ---------------------------------------------------------------------------
# _merge_batch_detections unit tests
# ---------------------------------------------------------------------------


class TestMergeBatchDetections:
    def test_single_detector_single_frame(self):
        fd = FrameDetections(
            frame_index=0,
            detections=[Detection(BoundingBox(0, 0, 10, 10), DetectionClass.PERSON, 0.9)],
        )
        result = _merge_batch_detections([[fd]], batch_len=1, start_frame_index=0)
        assert len(result) == 1
        assert len(result[0].detections) == 1

    def test_two_detectors_detections_merged_into_one_fd(self):
        fd1 = FrameDetections(
            frame_index=0,
            detections=[Detection(BoundingBox(0, 0, 10, 10), DetectionClass.PERSON, 0.9)],
        )
        fd2 = FrameDetections(
            frame_index=0,
            detections=[Detection(BoundingBox(20, 20, 30, 30), DetectionClass.LICENSE_PLATE, 0.8)],
        )
        result = _merge_batch_detections([[fd1], [fd2]], batch_len=1, start_frame_index=0)
        assert len(result[0].detections) == 2

    def test_no_detectors_returns_empty_fds(self):
        result = _merge_batch_detections([], batch_len=3, start_frame_index=10)
        assert len(result) == 3
        assert all(len(fd.detections) == 0 for fd in result)

    def test_frame_indices_assigned_from_start(self):
        result = _merge_batch_detections([], batch_len=3, start_frame_index=5)
        assert result[0].frame_index == 5
        assert result[1].frame_index == 6
        assert result[2].frame_index == 7

    def test_batch_of_four_frames(self):
        fds = [FrameDetections(frame_index=i) for i in range(4)]
        result = _merge_batch_detections([fds], batch_len=4, start_frame_index=0)
        assert len(result) == 4

    def test_detections_from_multiple_detectors_all_present(self):
        def make_fd(cls, frame_index=0):
            return FrameDetections(
                frame_index=frame_index,
                detections=[Detection(BoundingBox(0, 0, 1, 1), cls, 0.5)],
            )

        result = _merge_batch_detections(
            [
                [make_fd(DetectionClass.PERSON)],
                [make_fd(DetectionClass.LICENSE_PLATE)],
                [make_fd(DetectionClass.LOGO)],
            ],
            batch_len=1,
            start_frame_index=0,
        )
        classes = {d.detection_class for d in result[0].detections}
        assert DetectionClass.PERSON in classes
        assert DetectionClass.LICENSE_PLATE in classes
        assert DetectionClass.LOGO in classes


# ---------------------------------------------------------------------------
# CensorPipeline.build tests
# ---------------------------------------------------------------------------


class TestCensorPipelineBuild:
    @pytest.fixture
    def cfg(self, tmp_path):
        return AppConfig(
            io=IOConfig(
                input_path=str(tmp_path / "in.mp4"),
                output_path=str(tmp_path / "out.mp4"),
            )
        )

    def test_build_with_all_enabled_creates_three_detectors(self, cfg, mocker):
        mocker.patch("video_censor.detectors.person.YOLO")
        mocker.patch("video_censor.detectors.license_plate.hf_pipeline")
        mocker.patch("video_censor.detectors.logo.hf_pipeline")
        pipeline = CensorPipeline(cfg)
        pipeline.build()
        assert len(pipeline._detectors) == 3

    def test_build_persons_only(self, cfg, mocker):
        cfg.censor_license_plates = False
        cfg.censor_logos = False
        mocker.patch("video_censor.detectors.person.YOLO")
        pipeline = CensorPipeline(cfg)
        pipeline.build()
        assert len(pipeline._detectors) == 1
        assert pipeline._detectors[0].detection_class == DetectionClass.PERSON

    def test_build_no_detectors_when_all_disabled(self, cfg):
        cfg.censor_persons = False
        cfg.censor_license_plates = False
        cfg.censor_logos = False
        pipeline = CensorPipeline(cfg)
        pipeline.build()
        assert pipeline._detectors == []

    def test_build_creates_interactive_tracker_when_selections_present(self, cfg, mocker):
        from video_censor.models import BoundingBox, InteractiveSelection

        cfg.censor_persons = False
        cfg.censor_license_plates = False
        cfg.censor_logos = False
        cfg.interactive_selections = [
            InteractiveSelection(frame_index=0, bbox=BoundingBox(0, 0, 10, 10))
        ]
        pipeline = CensorPipeline(cfg)
        pipeline.build()
        assert pipeline._interactive_tracker is not None

    def test_build_no_interactive_tracker_without_selections(self, cfg, mocker):
        cfg.censor_persons = False
        cfg.censor_license_plates = False
        cfg.censor_logos = False
        pipeline = CensorPipeline(cfg)
        pipeline.build()
        assert pipeline._interactive_tracker is None


# ---------------------------------------------------------------------------
# CensorPipeline.run integration tests (all I/O and models mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_cfg(tmp_path):
    return AppConfig(
        io=IOConfig(
            input_path=str(tmp_path / "in.mp4"),
            output_path=str(tmp_path / "out.mp4"),
            batch_size=2,
        ),
        censor=CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0),
        censor_persons=False,
        censor_license_plates=False,
        censor_logos=False,
    )


def _make_mock_reader(mocker, total_frames=4, batch_size=2):
    """Build a mock VideoReader that returns `total_frames` dummy frames."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    frames_per_batch = [dummy] * batch_size
    n_full_batches = total_frames // batch_size
    batches = [frames_per_batch] * n_full_batches + [[]]

    mock_reader = mocker.MagicMock()
    mock_reader.__enter__ = lambda s: s
    mock_reader.__exit__ = mocker.MagicMock(return_value=False)
    mock_reader.metadata.total_frames = total_frames
    mock_reader.metadata.fps = 30.0
    mock_reader.metadata.width = 640
    mock_reader.metadata.height = 480
    mock_reader.metadata.fourcc = 0
    mock_reader.read_batch.side_effect = batches
    return mock_reader


def _make_mock_writer(mocker):
    mock_writer = mocker.MagicMock()
    mock_writer.__enter__ = lambda s: s
    mock_writer.__exit__ = mocker.MagicMock(return_value=False)
    return mock_writer


class TestCensorPipelineRun:
    def test_processes_all_frames(self, mocker, minimal_cfg):
        mock_reader = _make_mock_reader(mocker, total_frames=4, batch_size=2)
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        pipeline = CensorPipeline(minimal_cfg)
        pipeline.build()
        stats = pipeline.run()

        assert stats.processed_frames == 4

    def test_writes_correct_number_of_frames(self, mocker, minimal_cfg):
        mock_reader = _make_mock_reader(mocker, total_frames=4, batch_size=2)
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        pipeline = CensorPipeline(minimal_cfg)
        pipeline.build()
        pipeline.run()

        assert mock_writer.write_frame.call_count == 4

    def test_progress_callback_called_for_each_frame(self, mocker, minimal_cfg):
        mock_reader = _make_mock_reader(mocker, total_frames=4, batch_size=2)
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        calls = []
        pipeline = CensorPipeline(minimal_cfg)
        pipeline.build()
        pipeline.run(on_progress=lambda cur, tot: calls.append((cur, tot)))

        assert len(calls) == 4
        assert calls[-1][0] == 4

    def test_stats_elapsed_and_fps_populated(self, mocker, minimal_cfg):
        mock_reader = _make_mock_reader(mocker, total_frames=2, batch_size=2)
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        pipeline = CensorPipeline(minimal_cfg)
        pipeline.build()
        stats = pipeline.run()

        assert stats.elapsed_seconds >= 0
        assert stats.fps >= 0

    def test_empty_video_returns_zero_processed(self, mocker, minimal_cfg):
        mock_reader = mocker.MagicMock()
        mock_reader.__enter__ = lambda s: s
        mock_reader.__exit__ = mocker.MagicMock(return_value=False)
        mock_reader.metadata.total_frames = 0
        mock_reader.metadata.fps = 30.0
        mock_reader.metadata.width = 640
        mock_reader.metadata.height = 480
        mock_reader.metadata.fourcc = 0
        mock_reader.read_batch.return_value = []
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        pipeline = CensorPipeline(minimal_cfg)
        pipeline.build()
        stats = pipeline.run()

        assert stats.processed_frames == 0
        assert mock_writer.write_frame.call_count == 0

    def test_censored_regions_counted(self, mocker, minimal_cfg):
        from video_censor.models import BoundingBox, Detection, DetectionClass, FrameDetections

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_reader = mocker.MagicMock()
        mock_reader.__enter__ = lambda s: s
        mock_reader.__exit__ = mocker.MagicMock(return_value=False)
        mock_reader.metadata.total_frames = 1
        mock_reader.metadata.fps = 30.0
        mock_reader.metadata.width = 640
        mock_reader.metadata.height = 480
        mock_reader.metadata.fourcc = 0
        mock_reader.read_batch.side_effect = [[dummy], []]
        mock_writer = _make_mock_writer(mocker)
        mocker.patch("video_censor.pipeline.VideoReader", return_value=mock_reader)
        mocker.patch("video_censor.pipeline.VideoWriter", return_value=mock_writer)

        # Inject a mock detector that returns 2 detections per frame
        mock_detector = mocker.MagicMock()
        mock_detector.detection_class = DetectionClass.PERSON
        mock_detector.detect_batch.return_value = [
            FrameDetections(
                frame_index=0,
                detections=[
                    Detection(BoundingBox(0, 0, 10, 10), DetectionClass.PERSON, 0.9),
                    Detection(BoundingBox(20, 20, 30, 30), DetectionClass.PERSON, 0.8),
                ],
            )
        ]
        mock_detector.warmup.return_value = None

        pipeline = CensorPipeline(minimal_cfg)
        pipeline._detectors = [mock_detector]
        stats = pipeline.run()

        assert stats.total_censored_regions == 2
