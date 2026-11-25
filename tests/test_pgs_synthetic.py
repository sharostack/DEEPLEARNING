import torch
from torch import nn

from pgs.soft_argmax import soft_argmax_2d
from pgs.sips import calculate_sips
from pgs.proportion_loss import confidence_weighted_proportion_loss
from pgs.latent_correction import normalized_masked_correction
from pgs.pgs_controller import PGSConfig
from pgs.objective import PGSObjective
from pgs.vitpose import FrozenViTPose
from pgs.scheduler import ddim_pred_original_sample


def test_soft_argmax_localizes_peak_and_propagates_gradient():
    h = torch.zeros(1, 1, 5, 7, requires_grad=True)
    with torch.no_grad():
        h[0, 0, 2, 5] = 8
    xy, conf = soft_argmax_2d(h, beta=1.0)
    assert torch.allclose(xy[0, 0], torch.tensor([5 / 6, 2 / 4]), atol=0.1)
    (xy.sum() + conf.sum()).backward()
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_generic_sips_confidence_filter_and_zero_denominator():
    kp = torch.tensor([[[0., 0.], [3., 0.], [0., 0.], [0., 4.]]])
    c = torch.ones(1, 4)
    values, valid = calculate_sips(kp, c, {"r": (0, 1, 2, 3)})
    assert torch.allclose(values["r"], torch.tensor([0.75])) and valid["r"].item()
    c[0, 1] = 0.1
    _, valid = calculate_sips(kp, c, {"r": (0, 1, 2, 3)}, min_confidence=0.2)
    assert not valid["r"].item()


def test_loss_skips_invalid_ratios():
    loss, count = confidence_weighted_proportion_loss(
        {"r": torch.tensor([1.], requires_grad=True)}, {"r": torch.tensor([2.])},
        {"r": torch.tensor([False])})
    assert loss is None and count == 0


def test_latent_rms_normalization_mask_and_preservation():
    # CatVTON latent is [person rows, garment rows] in height, not channels.
    z = torch.ones(1, 2, 4, 2, requires_grad=True)
    loss = z.sum()
    mask = torch.tensor([[[1., 0.], [0., 0.]]])
    corrected, grad = normalized_masked_correction(z, loss, mask, strength=0.1,
                                                    person_height=2)
    assert torch.equal(corrected[:, :, 2:, :], z.detach()[:, :, 2:, :])
    assert torch.equal(corrected[:, :, 0, 1:], z.detach()[:, :, 0, 1:])
    assert torch.equal(corrected[:, :, 1, :], z.detach()[:, :, 1, :])
    assert not corrected.requires_grad
    assert torch.equal(grad[:, :, 2:, :], torch.zeros_like(grad[:, :, 2:, :]))
    expected = 1.0 - 0.1 / (2.0 ** 0.5)
    assert torch.allclose(corrected[0, 0, 0, 0], torch.tensor(expected), atol=1e-5)


def test_soft_argmax_marks_nonfinite_joint_invalid():
    heatmaps = torch.zeros(1, 1, 2, 2)
    heatmaps[0, 0, 0, 0] = float("nan")
    xy, confidence = soft_argmax_2d(heatmaps)
    assert torch.equal(xy, torch.zeros_like(xy))
    assert torch.equal(confidence, torch.zeros_like(confidence))


def test_sips_zero_denominator_returns_invalid_finite_value():
    points = torch.zeros(1, 4, 2)
    ratios, valid = calculate_sips(points, torch.ones(1, 4), {"r": (0, 1, 2, 3)})
    assert ratios["r"].item() == 0
    assert torch.isfinite(ratios["r"]).all() and not valid["r"].item()


def test_weighted_loss_has_finite_gradients():
    pred = torch.tensor([1.5, 99.], requires_grad=True)
    loss, count = confidence_weighted_proportion_loss(
        {"r": pred}, {"r": torch.tensor([1., float("nan")])},
        {"r": torch.tensor([True, True])}, {"r": torch.tensor([0.8, 0.9])})
    loss.backward()
    assert count == 1 and torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all() and pred.grad[1] == 0


def test_frozen_parameters_receive_no_gradient():
    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(False)
    latent = torch.ones(1, 2, requires_grad=True)
    loss = frozen(latent).square().sum()
    loss.backward()
    assert latent.grad is not None
    assert all(p.grad is None for p in frozen.parameters())


class _DummyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 2, 1, bias=False)
        self.keypoint_head = nn.Conv2d(2, 4, 1, bias=False)
        self.with_neck = False


def test_frozen_vitpose_wrapper_keeps_input_gradient():
    model = _DummyPose()
    pose = FrozenViTPose(model, input_size=(4, 4), expected_image_hw=(8, 8))
    rgb = torch.rand(1, 3, 8, 8, requires_grad=True)
    heatmaps = pose(rgb)
    heatmaps.sum().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in pose.parameters())


def test_pgs_objective_gradients_flow_through_frozen_pose():
    class CoordinatePose(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.requires_grad_(False)
        def forward(self, rgb):
            # Four joints; RGB's first channel shifts one joint's heatmap.
            h = rgb.new_zeros((rgb.shape[0], 4, 2, 2))
            h[:, 0, 0, 0] = rgb[:, 0].mean((1, 2)) * self.weight
            h[:, 0, 0, 1] = 1
            h[:, 1, 1, 1] = 2
            h[:, 2, 0, 0] = 2
            h[:, 3, 1, 0] = 2
            return h
    config = PGSConfig(enabled=True, beta=1., min_confidence=0.)
    objective = PGSObjective(CoordinatePose(), {"r": (0, 1, 2, 3)}, config)
    reference = torch.zeros(1, 3, 4, 4)
    objective.set_reference(reference)
    generated = torch.full((1, 3, 4, 4), .3, requires_grad=True)
    loss, count = objective(generated)
    assert loss is not None and count == 1
    loss.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert objective.pose_model.weight.grad is None


def test_ddim_x0_prediction_types():
    class Scheduler:
        alphas_cumprod = torch.tensor([0.25])
        class config:
            prediction_type = "epsilon"
            thresholding = False
            clip_sample = False
            clip_sample_range = 1.0
    scheduler = Scheduler()
    sample, output = torch.tensor([[[[2.]]]]), torch.tensor([[[[1.]]]])
    expected_eps = (2. - (0.75 ** 0.5)) / 0.5
    assert torch.allclose(ddim_pred_original_sample(scheduler, output, 0, sample), torch.tensor([[[[expected_eps]]]]))
    scheduler.config.prediction_type = "sample"
    assert torch.equal(ddim_pred_original_sample(scheduler, output, 0, sample), output)
    scheduler.config.prediction_type = "v_prediction"
    expected_v = 0.5 * 2. - (0.75 ** 0.5)
    assert torch.allclose(ddim_pred_original_sample(scheduler, output, 0, sample), torch.tensor([[[[expected_v]]]]))


def test_correction_schedule_and_experiment_enablement():
    configured = PGSConfig(enabled=True, forward_iteration_indices=(2, 5),
                           correction_strength=0.03, strength_by_step={5: 0.07})
    assert not configured.should_correct(1)
    assert configured.should_correct(2)
    assert configured.should_correct(5)
    assert configured.strength_at(2) == 0.03
    assert configured.strength_at(5) == 0.07
    assert not PGSConfig(enabled=False).should_correct(2)
