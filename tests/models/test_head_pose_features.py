import math

import torch

from project.models.head_pose_features import (
    FEATURE_DIM,
    HEAD_POINT_INDICES,
    LEAR,
    NOSE,
    REAR,
    head_pose_features,
)


def _skeleton(yaw_deg: float = 0.0, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """A synthetic 70-point frame: the whole head rotated by yaw around y.

    Ears must rotate together with the nose -- head orientation is defined
    relative to the ear axis, so rotating only the nose is not a head turn.
    """
    pts = torch.zeros(70, 3)
    a = math.radians(yaw_deg)
    rot = torch.tensor(
        [
            [math.cos(a), 0.0, math.sin(a)],
            [0.0, 1.0, 0.0],
            [-math.sin(a), 0.0, math.cos(a)],
        ]
    )
    pts[LEAR] = rot @ torch.tensor([0.1, 0.0, 0.0])
    pts[REAR] = rot @ torch.tensor([-0.1, 0.0, 0.0])
    pts[NOSE] = rot @ torch.tensor([0.0, 0.0, 0.15])
    for i in HEAD_POINT_INDICES:
        pts[i] += torch.tensor(offset)
    return pts


def test_shape_and_dim():
    kpts = torch.randn(2, 8, 70, 3)
    out = head_pose_features(kpts)
    assert out.shape == (2, 8, FEATURE_DIM)


def test_yaw_recovers_synthetic_rotation():
    frames = torch.stack([_skeleton(0.0), _skeleton(30.0), _skeleton(-30.0)])
    feats = head_pose_features(frames.unsqueeze(0), center=False)[0]
    yaw = torch.rad2deg(feats[:, 0])
    assert abs(yaw[0]) < 1.0
    assert abs(yaw[1] - 30.0) < 1.0
    assert abs(yaw[2] + 30.0) < 1.0


def test_translation_invariance_of_angles_and_local_coords():
    """Moving the whole head (different seat position / driver height) must not
    change anything except the centroid-velocity channels."""
    a = head_pose_features(
        torch.stack([_skeleton(10.0), _skeleton(20.0)]).unsqueeze(0), center=False
    )
    b = head_pose_features(
        torch.stack(
            [_skeleton(10.0, offset=(0.5, -0.3, 0.2)), _skeleton(20.0, offset=(0.5, -0.3, 0.2))]
        ).unsqueeze(0),
        center=False,
    )
    keep = [i for i in range(FEATURE_DIM) if i not in (6, 7, 8)]  # drop centroid vel
    assert torch.allclose(a[..., keep], b[..., keep], atol=1e-5)


def test_centering_zeroes_the_temporal_mean():
    kpts = torch.randn(1, 12, 70, 3)
    feats = head_pose_features(kpts, center=True)
    angles_mean = feats[..., :3].mean(dim=-2)
    local_mean = feats[..., 9:].mean(dim=-2)
    assert angles_mean.abs().max() < 1e-5
    assert local_mean.abs().max() < 1e-5


def test_delta_channels_catch_a_movement_step():
    still = _skeleton(0.0)
    moved = _skeleton(25.0)
    seq = torch.stack([still, still, moved, moved]).unsqueeze(0)
    feats = head_pose_features(seq, center=False)[0]
    d_yaw = feats[:, 3]
    assert d_yaw[2].abs() > math.radians(20)  # the step frame
    assert d_yaw[1].abs() < 1e-4 and d_yaw[3].abs() < 1e-4
