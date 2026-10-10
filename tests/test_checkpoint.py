import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from zesyde.checkpoint import PauseController, TrainingInterruptedException

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        controller = PauseController(checkpoint_dir=tmp_path, enable_interactive=False, enable_signals=False)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Do a dummy forward/backward to populate optimizer state
        x = torch.randn(2, 4)
        out = model(x).sum()
        out.backward()
        optimizer.step()
        scheduler.step()

        # Save state
        saved_path = controller.save_state(
            phase="phase_1_warmstart",
            epoch=2,
            total_epochs=5,
            step=42,
            total_steps=100,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state={"temperatures": {"(choice,4)": 1.15}},
        )

        assert saved_path.exists()

        # Load state
        ckpt = PauseController.load_state(saved_path)
        assert ckpt["phase"] == "phase_1_warmstart"
        assert ckpt["epoch"] == 2
        assert ckpt["step"] == 42
        assert ckpt["extra_state"]["temperatures"]["(choice,4)"] == 1.15

        # Verify model parameters match
        new_model = SimpleModel()
        new_model.load_state_dict(ckpt["model_state_dict"])
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

def test_sentinel_file_pause_trigger():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        controller = PauseController(checkpoint_dir=tmp_path, enable_interactive=False, enable_signals=False)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create sentinel file
        controller.sentinel_file.touch()
        assert controller.sentinel_file.exists()

        # check_pause in non-interactive environment should trigger save & raise TrainingInterruptedException
        with pytest.raises(TrainingInterruptedException) as exc_info:
            controller.check_pause(
                phase="phase_1_warmstart",
                epoch=1,
                total_epochs=3,
                step=5,
                total_steps=50,
                model=model,
                optimizer=optimizer,
            )

        assert "sentinel file" in str(exc_info.value)
        assert exc_info.value.checkpoint_path.exists()

def test_signal_interrupted_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        controller = PauseController(checkpoint_dir=tmp_path, enable_interactive=False, enable_signals=False)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate SIGINT flag
        controller.interrupted = True

        with pytest.raises(TrainingInterruptedException) as exc_info:
            controller.check_pause(
                phase="phase_1_warmstart",
                epoch=1,
                total_epochs=3,
                step=10,
                total_steps=50,
                model=model,
                optimizer=optimizer,
            )

        assert "Ctrl+C signal" in str(exc_info.value)
        assert exc_info.value.checkpoint_path.exists()
