import numpy as np
import pytest

from video_censor.io.video_reader import VideoMetadata, VideoReader


class TestVideoMetadata:
    def test_is_frozen_dataclass(self):
        meta = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=300, fourcc=0)
        with pytest.raises((TypeError, AttributeError)):
            meta.width = 640  # type: ignore[misc]

    def test_stores_all_fields(self):
        meta = VideoMetadata(width=640, height=480, fps=25.0, total_frames=100, fourcc=1)
        assert meta.width == 640
        assert meta.height == 480
        assert meta.fps == 25.0
        assert meta.total_frames == 100
        assert meta.fourcc == 1


class TestVideoReader:
    def test_open_nonexistent_file_raises_file_not_found(self, tmp_path):
        reader = VideoReader(str(tmp_path / "nonexistent.mp4"))
        with pytest.raises(FileNotFoundError, match="Cannot open video"):
            reader.open()

    def test_context_manager_opens_and_closes(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        with VideoReader("/fake/path.mp4"):
            pass
        mock_cap.release.assert_called_once()

    def test_read_frame_returns_frame_when_available(self, mocker):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy)
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        frame = reader.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_read_frame_returns_none_at_eof(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        assert reader.read_frame() is None

    def test_read_batch_returns_correct_count(self, mocker):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy)
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        batch = reader.read_batch(4)
        assert len(batch) == 4

    def test_read_batch_stops_at_eof(self, mocker):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, dummy),
            (True, dummy),
            (False, None),
        ]
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        batch = reader.read_batch(8)
        assert len(batch) == 2

    def test_read_batch_empty_at_eof_immediately(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        assert reader.read_batch(8) == []

    def test_assert_open_raises_when_not_opened(self):
        reader = VideoReader("/fake/path.mp4")
        with pytest.raises(RuntimeError, match="not open"):
            reader.read_frame()

    def test_assert_open_raises_on_metadata_when_not_opened(self):
        reader = VideoReader("/fake/path.mp4")
        with pytest.raises(RuntimeError, match="not open"):
            _ = reader.metadata

    def test_metadata_returns_correct_values(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True

        def get_side_effect(prop):
            return {
                0: 1280,   # CAP_PROP_FRAME_WIDTH
                1: 720,    # CAP_PROP_FRAME_HEIGHT
                5: 29.97,  # CAP_PROP_FPS
                7: 500,    # CAP_PROP_FRAME_COUNT
                6: 828601953,  # CAP_PROP_FOURCC
            }.get(prop, 0)

        mock_cap.get.side_effect = get_side_effect
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)

        import cv2 as _cv2
        # Patch the constants used in metadata
        mocker.patch.object(_cv2, "CAP_PROP_FRAME_WIDTH", 0)
        mocker.patch.object(_cv2, "CAP_PROP_FRAME_HEIGHT", 1)
        mocker.patch.object(_cv2, "CAP_PROP_FPS", 5)
        mocker.patch.object(_cv2, "CAP_PROP_FRAME_COUNT", 7)
        mocker.patch.object(_cv2, "CAP_PROP_FOURCC", 6)

        reader = VideoReader("/fake/path.mp4")
        reader.open()
        meta = reader.metadata
        assert meta.width == 1280
        assert meta.height == 720

    def test_seek_calls_set_on_cap(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        reader.seek(42)
        import cv2 as _cv2
        mock_cap.set.assert_called_with(_cv2.CAP_PROP_POS_FRAMES, 42)

    def test_read_frame_at_restores_position(self, mocker):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy)
        mock_cap.get.return_value = 10  # current position before seek
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        reader.read_frame_at(5)
        # Should have sought back to saved position (10)
        import cv2 as _cv2
        set_calls = [c for c in mock_cap.set.call_args_list
                     if c[0][0] == _cv2.CAP_PROP_POS_FRAMES]
        positions = [c[0][1] for c in set_calls]
        assert 5 in positions
        assert 10 in positions

    def test_close_is_idempotent(self, mocker):
        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        reader = VideoReader("/fake/path.mp4")
        reader.open()
        reader.close()
        reader.close()  # should not raise
        mock_cap.release.assert_called_once()
