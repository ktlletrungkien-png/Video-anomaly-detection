import pytest

from rwf2000.sampling import sample_uniform_indices


def test_uniform_sampler_preserves_temporal_order_and_half_open_boundary():
    indices = sample_uniform_indices(100, 16, start=10, stop=30)
    assert len(indices) == 16
    assert indices[0] == 10
    assert indices[-1] == 29
    assert indices == sorted(indices)
    assert all(10 <= index < 30 for index in indices)


def test_sampler_rejects_short_clip_without_padding_or_crossing_boundary():
    with pytest.raises(ValueError, match="strict policy"):
        sample_uniform_indices(15, 16)
    with pytest.raises(ValueError, match="strict policy"):
        sample_uniform_indices(40, 16, start=8, stop=23)
