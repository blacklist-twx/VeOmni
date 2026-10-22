"""H3 prefix decoding preserves sampled frames without reading an unused tail."""

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.multimodal.dit import data_transform


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class VideoReader:
    def __init__(self, fps, count, duration=100, fail_at=None):
        self.fps, self.count, self.duration = fps, count, duration
        self.fail_at = fail_at
        self.decoded = 0
        self.closed = False

    def get_meta_data(self):
        return {"fps": self.fps, "duration": self.duration}

    def iter_data(self):
        for i in range(self.count):
            if i == self.fail_at:
                raise RuntimeError("decode failed")
            self.decoded += 1
            yield np.full((4, 4, 3), i % 256, dtype=np.uint8)

    def close(self):
        self.closed = True


def load(monkeypatch, reader):
    import imageio

    monkeypatch.setattr(imageio, "get_reader", lambda _: reader)
    monkeypatch.setattr(data_transform, "conv_preprocess", lambda **_: ("prompt", {}, [], ["clip.mp4"]))
    return data_transform.process_minimax_h3_online_example({}, "unused", min_frames=22, fps=24, height=4, width=4)[0]


@pytest.mark.parametrize("raw_fps", [12, 24, 30, 60])
def test_decode_stops_at_last_requested_frame(monkeypatch, raw_fps):
    reader = VideoReader(raw_fps, count=2400)
    sample = load(monkeypatch, reader)
    desired = [round(i / 24 * raw_fps) for i in range(22)]
    assert reader.decoded == desired[-1] + 1
    assert reader.closed
    assert len(sample["videos"]) == 22
    for frame, index in zip(sample["videos"], desired):
        torch.testing.assert_close(frame, torch.full((3, 4, 4), index / 255), rtol=0, atol=1e-7)
    assert np.array(sample["images"][0]).mean() == desired[0]
    assert np.array(sample["images"][1]).mean() == desired[-1]


def test_truncated_stream_still_repeats_last_frame(monkeypatch):
    reader = VideoReader(24, count=17)
    sample = load(monkeypatch, reader)
    assert reader.decoded == 17 and reader.closed
    assert len(sample["videos"]) == 22
    for frame in sample["videos"][16:]:
        torch.testing.assert_close(frame, sample["videos"][16], rtol=0, atol=0)


def test_reader_closes_on_decode_error(monkeypatch):
    reader = VideoReader(24, count=100, fail_at=3)
    with pytest.raises(RuntimeError, match="decode failed"):
        load(monkeypatch, reader)
    assert reader.closed
