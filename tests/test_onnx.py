import pytest
import torch
from zesyde.onnx_export import export_model

def test_onnx_export_signature():
    # Ensure the function signature is correct and callable
    assert callable(export_model)

