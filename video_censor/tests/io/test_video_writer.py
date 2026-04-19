import numpy as np
import pytest

from video_censor.io.video_reader import VideoMetadata
from video_censor.io.video_writer import VideoWriter


@pytest.fixture
def meta() -> VideoMetadata:
    return VideoMetadata(width=640, height=480, fps=30.0, total_frames=100, fourcc=0)


class TestVideoWriter:
    def test_open_failure_raises_runtime_error(self, mocker, meta, tmp_path):
        mock_writer = mocker.MagicMock()
        mock_writer.isOpened.return_value = False
        mocker.patch("cv2.VideoWriter", return_value=mock_writer)
        mocker.patch("cv2.VideoWriter_fourcc", return_value=0)
        writer = VideoWriter(str(tmp_path / "out.mp4"), meta)
        with pytest.raises(RuntimeError, match="Cannot open output video"):
            writer.open()

    def test_context_manager_opens_and_closes(self, mocker, meta, tmp_path):
        mock_writer = mocker.MagicMock()
        mock_writer.isOpened.return_value = True
        mocker.patch("cv2.VideoWriter", return_value=mock_writer)
        mocker.patch("cv2.VideoWriter_fourcc", return_value=0)
        with VideoWriter(str(tmp_path / "out.mp4"), meta):
            pass
        mock_writer.release.assert_called_once()

    def test_write_frame_calls_underlying_writer(self, mocker, meta, tmp_path):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_writer = mocker.MagicMock()
        mock_writer.isOpened.return_value = True
        mocker.patch("cv2.VideoWriter", return_value=mock_writer)
        mocker.patch("cv2.VideoWriter_fourcc", return_value=0)
        writer = VideoWriter(str(tmp_path / "out.mp4"), meta)
        writer.open()
        writer.write_frame(dummy)
        mock_writer.write.assert_called_once_with(dummy)

    def test_write_frame_raises_when_not_open(self, meta, tmp_path):
        writer = VideoWriter(str(tmp_path / "out.mp4"), meta)
        with pytest.raises(RuntimeError, match="not open"):
            writer.write_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_close_is_idempotent(self, mocker, meta, tmp_path):
        mock_writer = mocker.MagicMock()
        mock_writer.isOpened.return_value = True
        mocker.patch("cv2.VideoWriter", return_value=mock_writer)
        mocker.patch("cv2.VideoWriter_fourcc", return_value=0)
        writer = VideoWriter(str(tmp_path / "out.mp4"), meta)
        writer.open()
        writer.close()
        writer.close()  # should not raise
        mock_writer.release.assert_called_once()

    def test_uses_mp4v_fourcc(self, mocker, meta, tmp_path):
        mock_writer = mocker.MagicMock()
        mock_writer.isOpened.return_value = True
        mock_fourcc = mocker.patch("cv2.VideoWriter_fourcc", return_value=828601953)
        mocker.patch("cv2.VideoWriter", return_value=mock_writer)
        writer = VideoWriter(str(tmp_path / "out.mp4"), meta)
        writer.open()
        mock_fourcc.assert_called_once_with(*"mp4v")
