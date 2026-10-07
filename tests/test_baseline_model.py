import pytest
import torch
from torch import nn

from rwf2000.models import ResNet18TemporalAverage


class MeanFeatureBackbone(nn.Module):
    feature_dim = 2

    def forward(self, frames):
        return torch.stack(
            (frames[:, 0].mean(dim=(1, 2)), frames[:, 1].mean(dim=(1, 2))),
            dim=1,
        )


class BatchNormFeatureBackbone(nn.Module):
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, frames):
        return self.features(frames)


def _model(*, feature_dim=2, freeze_backbone=False):
    backbone = MeanFeatureBackbone()
    return ResNet18TemporalAverage(
        weights=None,
        backbone=backbone,
        feature_dim=feature_dim,
        freeze_backbone=freeze_backbone,
    )


def test_forward_keeps_batch_dimension_and_returns_finite_logits():
    model = _model()

    for batch_size in (1, 2):
        logits = model(torch.randn(batch_size, 3, 3, 8, 8))
        assert logits.shape == (batch_size,)
        assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "frames",
    [
        torch.randn(3, 3, 8, 8),
        torch.randn(2, 3, 1, 8, 8),
        torch.randn(2, 3, 3, 0, 8),
        torch.randn(2, 0, 3, 8, 8),
    ],
)
def test_forward_rejects_invalid_shapes(frames):
    model = _model()
    with pytest.raises(ValueError, match="frames"):
        model(frames)


def test_temporal_mean_is_applied_before_classifier():
    model = _model()
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.classifier.bias.zero_()

    frames = torch.zeros(1, 3, 3, 1, 1)
    frames[0, :, 0, 0, 0] = torch.tensor([1.0, 3.0, 5.0])

    logits = model(frames)

    assert torch.allclose(logits, torch.tensor([3.0]))


def test_frozen_backbone_is_not_trainable_and_stays_in_eval_mode():
    backbone = BatchNormFeatureBackbone()
    model = ResNet18TemporalAverage(
        weights=None,
        backbone=backbone,
        feature_dim=4,
        freeze_backbone=True,
    )
    model.train()

    assert model.training
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_frozen_batchnorm_buffers_do_not_change_after_optimizer_step():
    model = ResNet18TemporalAverage(
        weights=None,
        backbone=BatchNormFeatureBackbone(),
        feature_dim=4,
        freeze_backbone=True,
    )
    model.train()
    running_mean = model.backbone.features[1].running_mean.detach().clone()
    running_var = model.backbone.features[1].running_var.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    frames = torch.randn(2, 3, 3, 8, 8)
    loss = nn.BCEWithLogitsLoss()(model(frames), torch.tensor([0.0, 1.0]))
    loss.backward()
    optimizer.step()

    assert torch.equal(running_mean, model.backbone.features[1].running_mean)
    assert torch.equal(running_var, model.backbone.features[1].running_var)


def test_bce_with_logits_is_finite_and_backpropagates_to_classifier():
    model = _model(freeze_backbone=True)
    logits = model(torch.randn(2, 4, 3, 8, 8))
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))

    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_real_torchvision_resnet18_weights_none_cpu_smoke():
    model = ResNet18TemporalAverage(weights=None, freeze_backbone=True)
    logits = model(torch.randn(1, 2, 3, 32, 32))

    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()
