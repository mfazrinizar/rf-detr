# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for XLA/TPU detection utilities."""

from unittest import mock

import pytest

from rfdetr.utilities.xla import is_torch_xla_available, is_xla_device


class TestIsTorchXlaAvailable:
    """Tests for is_torch_xla_available()."""

    def test_returns_bool(self):
        result = is_torch_xla_available()
        assert isinstance(result, bool)

    def test_returns_false_when_import_fails(self):
        # Clear the lru_cache so the mock takes effect.
        is_torch_xla_available.cache_clear()
        with mock.patch.dict("sys.modules", {"torch_xla": None}):
            assert is_torch_xla_available() is False
        is_torch_xla_available.cache_clear()

    def test_returns_true_when_import_succeeds(self):
        is_torch_xla_available.cache_clear()
        fake_xla = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"torch_xla": fake_xla}):
            assert is_torch_xla_available() is True
        is_torch_xla_available.cache_clear()


class TestIsXlaDevice:
    """Tests for is_xla_device()."""

    def test_none_returns_false(self):
        assert is_xla_device(None) is False

    def test_cpu_string_returns_false(self):
        assert is_xla_device("cpu") is False

    def test_cuda_string_returns_false(self):
        assert is_xla_device("cuda") is False

    def test_xla_string_returns_true(self):
        assert is_xla_device("xla") is True

    def test_xla_uppercase_returns_true(self):
        # Device type comparison is case-insensitive.
        assert is_xla_device("XLA") is True

    def test_torch_device_xla(self):
        """Object with .type attribute set to 'xla' is detected."""
        mock_device = mock.MagicMock()
        mock_device.type = "xla"
        assert is_xla_device(mock_device) is True

    def test_torch_device_cuda(self):
        mock_device = mock.MagicMock()
        mock_device.type = "cuda"
        assert is_xla_device(mock_device) is False
