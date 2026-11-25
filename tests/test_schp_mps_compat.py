"""Synthetic checks for the inference-only PyTorch SCHP ABN fallback."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


SCHP_ROOT = Path(__file__).resolve().parents[1] / "catvtoncode" / "Comfyui-CatVTON" / "model" / "SCHP"
for import_root in (SCHP_ROOT.parent, SCHP_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from modules.bn import InPlaceABN, InPlaceABNSync  # noqa: E402
from modules.functions import INPLACE_ABN_EXTENSION_AVAILABLE  # noqa: E402
from SCHP.networks.AugmentCE2P import MPSCompatibleAdaptiveAvgPool2d  # noqa: E402


def _test_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.mark.parametrize("module_type", [InPlaceABN, InPlaceABNSync])
@pytest.mark.parametrize("activation", ["none", "relu", "leaky_relu", "elu"])
def test_pytorch_abn_eval_matches_fused_forward_formula(module_type, activation):
    device = _test_device()
    module = module_type(4, eps=1e-5, activation=activation, slope=0.07).to(device).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([-1.2, -0.4, 0.7, 1.3], device=device))
        module.bias.copy_(torch.tensor([0.2, -0.1, 0.4, -0.3], device=device))
        module.running_mean.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4], device=device))
        module.running_var.copy_(torch.tensor([0.8, 1.1, 1.7, 0.5], device=device))
    x = torch.linspace(-2, 2, 2 * 4 * 7 * 5, device=device).reshape(2, 4, 7, 5)

    expected = F.batch_norm(
        x, module.running_mean, module.running_var, training=False, momentum=module.momentum,
        eps=module.eps,
    )
    expected = expected * (module.weight.abs() + module.eps).view(1, 4, 1, 1)
    expected = expected + module.bias.view(1, 4, 1, 1)
    if activation == "relu":
        expected = F.relu(expected)
    elif activation == "leaky_relu":
        expected = F.leaky_relu(expected, negative_slope=module.slope)
    elif activation == "elu":
        expected = F.elu(expected)

    actual = module(x)
    assert actual.device.type == device.type
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_fallback_preserves_checkpoint_parameter_and_buffer_names():
    ordinary = InPlaceABN(8)
    synchronized = InPlaceABNSync(8)
    assert ordinary.state_dict().keys() == synchronized.state_dict().keys()
    assert set(ordinary.state_dict()) == {"weight", "bias", "running_mean", "running_var"}


def test_pytorch_fallback_refuses_training_mode():
    module = InPlaceABNSync(4).train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        module(torch.randn(2, 4, 3, 3))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA extension path is intentionally retained")
def test_non_cuda_runtime_does_not_build_cuda_extension():
    assert INPLACE_ABN_EXTENSION_AVAILABLE is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_nondivisible_mps_adaptive_pool_uses_exact_cpu_operation():
    source = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    mps_input = source.to("mps")
    output = MPSCompatibleAdaptiveAvgPool2d((3, 3))(mps_input)
    expected = F.adaptive_avg_pool2d(source, (3, 3)).to("mps")
    assert output.device.type == "mps"
    assert torch.equal(output.cpu(), expected.cpu())
