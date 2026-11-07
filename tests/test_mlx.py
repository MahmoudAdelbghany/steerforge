from __future__ import annotations

import sys

import pytest

from steerforge.mlx_backend import mlx_available


@pytest.mark.skipif(sys.platform == "darwin", reason="MLX is expected on macOS")
def test_mlx_guard_on_non_apple():
    assert mlx_available() is False


@pytest.mark.skipif(sys.platform == "darwin", reason="MLX is expected on macOS")
def test_mlx_backend_error_is_clear():
    from steerforge.mlx_backend import MLXBackend

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        MLXBackend("mlx-community/anything")
