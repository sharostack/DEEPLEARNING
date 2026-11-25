"""Synthetic regression test for the vendored SCHP Lovasz loss."""
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (Path(__file__).resolve().parents[1] / "catvtoncode" /
               "Comfyui-CatVTON" / "model" / "SCHP" / "utils" / "lovasz_softmax.py")
spec = importlib.util.spec_from_file_location("schp_lovasz_under_test", MODULE_PATH)
lovasz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lovasz)


def test_present_classes_does_not_depend_on_string_interning():
    probabilities = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    labels = torch.tensor([0, 0])
    ordinary = lovasz.lovasz_softmax_flat(probabilities, labels, classes="present")
    dynamically_created = bytes(bytearray(b"present")).decode("ascii")
    assert dynamically_created == "present"
    dynamic = lovasz.lovasz_softmax_flat(probabilities, labels, classes=dynamically_created)
    assert torch.allclose(ordinary, dynamic)
