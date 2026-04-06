# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for TPU-related code paths in the training stack.

These tests run on CPU by mocking XLA availability — they verify that TPU
code paths are wired correctly without requiring actual TPU hardware.
"""

from unittest import mock

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig


def _mc(**kwargs):
    """Minimal RFDETRBaseConfig for tests."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kwargs)
    return RFDETRBaseConfig(**defaults)


def _tc(tmp_path, **kwargs):
    """Minimal TrainConfig for tests."""
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


class TestTrainerTPUPrecision:
    """build_trainer resolves precision correctly for TPU."""

    def test_tpu_precision_bf16_true(self, tmp_path):
        """When accelerator is 'tpu' and xla available, precision should be bf16-true."""
        from rfdetr.training.trainer import build_trainer

        mc = _mc(amp=True)
        tc = _tc(tmp_path)
        # We cannot construct a full Trainer(accelerator="tpu") without torch_xla.
        # Instead, build with accelerator="cpu" but mock is_torch_xla_available
        # and pass accelerator="tpu" — capture the precision kwarg via a mock.
        precision_values = []
        original_trainer_init = torch.nn.Module.__init__  # just a placeholder

        with mock.patch("rfdetr.training.trainer.is_torch_xla_available", return_value=True):
            with mock.patch("rfdetr.training.trainer.Trainer") as MockTrainer:
                MockTrainer.return_value = mock.MagicMock()
                build_trainer(tc, mc, accelerator="tpu")
                call_kwargs = MockTrainer.call_args[1]
                assert call_kwargs["precision"] == "bf16-true"

    def test_tpu_precision_fp32_when_amp_off(self, tmp_path):
        """When amp=False, precision should be 32-true regardless of accelerator."""
        from rfdetr.training.trainer import build_trainer

        mc = _mc(amp=False)
        tc = _tc(tmp_path)
        with mock.patch("rfdetr.training.trainer.is_torch_xla_available", return_value=True):
            with mock.patch("rfdetr.training.trainer.Trainer") as MockTrainer:
                MockTrainer.return_value = mock.MagicMock()
                build_trainer(tc, mc, accelerator="tpu")
                call_kwargs = MockTrainer.call_args[1]
                assert call_kwargs["precision"] == "32-true"


class TestModuleModelTPUGuards:
    """Module model disables CUDA-specific features on XLA."""

    def test_fused_optimizer_disabled_on_xla(self, tmp_path):
        """fused=False when device is XLA, even if fused_optimizer config is True."""
        from rfdetr.training.module_model import RFDETRModelModule

        mc = _mc(fused_optimizer=True)
        tc = _tc(tmp_path)
        module = RFDETRModelModule(mc, tc)

        # Mock trainer for configure_optimizers
        mock_trainer = mock.MagicMock()
        mock_trainer.estimated_stepping_batches = 100
        type(module).trainer = property(lambda self: mock_trainer)

        # Patch is_xla_device at the module level to return True
        with mock.patch("rfdetr.training.module_model.is_xla_device", return_value=True):
            opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]
        # fused should not be enabled
        for group in optimizer.param_groups:
            assert not group.get("fused", False), "fused should be False on XLA"


class TestModuleDataTPUGuards:
    """DataModule disables pin_memory on XLA."""

    def test_pin_memory_false_on_xla(self, tmp_path):
        """pin_memory should default to False when device is xla."""
        from rfdetr.training.module_data import RFDETRDataModule

        mc = _mc()
        mc.device = "xla"
        tc = _tc(tmp_path, pin_memory=None)
        dm = RFDETRDataModule(mc, tc)
        assert dm._pin_memory is False

    def test_pin_memory_explicit_true_respected(self, tmp_path):
        """Explicit pin_memory=True overrides XLA default."""
        from rfdetr.training.module_data import RFDETRDataModule

        mc = _mc()
        mc.device = "xla"
        tc = _tc(tmp_path, pin_memory=True)
        dm = RFDETRDataModule(mc, tc)
        assert dm._pin_memory is True


class TestDetrDeviceMapping:
    """RFDETR._resolve_trainer_device_kwargs maps xla correctly."""

    def test_xla_maps_to_tpu_accelerator(self):
        from rfdetr.detr import RFDETR

        # torch.device('xla') requires torch_xla; test with string parsing
        with mock.patch("torch.device") as mock_device:
            mock_dev = mock.MagicMock()
            mock_dev.type = "xla"
            mock_dev.index = None
            mock_device.return_value = mock_dev
            accel, devices = RFDETR._resolve_trainer_device_kwargs("xla")
        assert accel == "tpu"
        assert devices is None

    def test_auto_batch_raises_on_tpu(self):
        """batch_size='auto' should raise ValueError when accelerator is tpu."""
        from rfdetr.detr import RFDETR

        model = mock.MagicMock(spec=RFDETR)
        model.model_config = _mc()
        model.get_train_config = mock.MagicMock(
            return_value=_tc(
                pytest.importorskip("pathlib").Path("/tmp/test_tpu"),
                batch_size="auto",
            )
        )
        model.model = mock.MagicMock()

        with mock.patch.object(
            RFDETR, "_resolve_trainer_device_kwargs", return_value=("tpu", None)
        ):
            with pytest.raises(ValueError, match="not supported on TPU"):
                RFDETR.train(model, device="xla")
