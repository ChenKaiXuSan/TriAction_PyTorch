#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Analytic head-pose features from SAM-3D keypoints.

Why this exists (2026-07-29 review of the failed keypoint-guidance arms):
the raw 70x3 keypoint tensor fed to the earlier kpt streams is 60% hand
coordinates (42 of 70 rows, moving 4x more than the head), in unnormalised
camera coordinates that encode who/where the driver is rather than what the
head does. The head-orientation signal is strong -- yaw spans ~69 degrees over
a clip -- but it lives in the *relative configuration* of the head points,
which a small dataset cannot be expected to discover from raw coordinates.

So compute what the network was supposed to learn:

- keep only the 8 head-related points (nose, eyes, ears, acromions, neck);
- build a per-frame head-local frame from the ear axis and the nose direction,
  and read yaw/pitch/roll off it analytically;
- add first differences (the label is a *movement* direction -- a derivative)
  and the head-centroid velocity;
- express the head points in the local frame (shape, not position);
- optionally subtract the per-segment mean pose, so features encode the
  deviation within this segment: removes each driver's resting posture
  (identity) and keeps exactly the movement.

33 features per frame: 3 angles + 3 angle deltas + 3 centroid velocity +
8 points x 3 local coords.
"""

from __future__ import annotations

import torch

# raw SAM-3D rows: nose, left-eye, right-eye, left-ear, right-ear ... left/right
# acromion and neck (see KEEP_KEYPOINT_INDICES in project/map_config.py)
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
L_ACROMION, R_ACROMION, NECK = 67, 68, 69
HEAD_POINT_INDICES = (NOSE, LEYE, REYE, LEAR, REAR, L_ACROMION, R_ACROMION, NECK)

FEATURE_DIM = 3 + 3 + 3 + len(HEAD_POINT_INDICES) * 3


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def head_pose_features(kpts: torch.Tensor, center: bool = True) -> torch.Tensor:
    """(..., T, K>=70, 3) raw keypoints -> (..., T, FEATURE_DIM) pose features.

    ``center=True`` subtracts the temporal mean of the angles and local
    coordinates over the segment (velocities are differences already).
    """
    if kpts.shape[-1] != 3:
        raise ValueError(f"Expected (..., T, K, 3) keypoints, got {tuple(kpts.shape)}")
    if kpts.shape[-2] <= max(HEAD_POINT_INDICES):
        raise ValueError(
            f"Need at least {max(HEAD_POINT_INDICES) + 1} keypoint rows, "
            f"got {kpts.shape[-2]}"
        )
    kpts = kpts.float()
    head = kpts[..., list(HEAD_POINT_INDICES), :]  # (..., T, 8, 3)

    ears_mid = (kpts[..., LEAR, :] + kpts[..., REAR, :]) / 2.0
    ear_axis = _normalize(kpts[..., LEAR, :] - kpts[..., REAR, :])
    forward = kpts[..., NOSE, :] - ears_mid
    forward = _normalize(
        forward - (forward * ear_axis).sum(-1, keepdim=True) * ear_axis
    )
    up = _normalize(torch.cross(ear_axis, forward, dim=-1))

    yaw = torch.atan2(forward[..., 0], forward[..., 2].abs().clamp_min(1e-6))
    pitch = torch.atan2(
        forward[..., 1], forward[..., [0, 2]].norm(dim=-1).clamp_min(1e-6)
    )
    roll = torch.atan2(
        ear_axis[..., 1], ear_axis[..., [0, 2]].norm(dim=-1).clamp_min(1e-6)
    )
    angles = torch.stack([yaw, pitch, roll], dim=-1)  # (..., T, 3)

    def _tdiff(x: torch.Tensor) -> torch.Tensor:
        d = torch.diff(x, dim=-2)
        return torch.cat([torch.zeros_like(d[..., :1, :]), d], dim=-2)

    d_angles = _tdiff(angles)
    centroid_vel = _tdiff(head.mean(dim=-2))

    # head points in the local frame: rows of R are the frame axes
    rot = torch.stack([ear_axis, forward, up], dim=-2)  # (..., T, 3, 3)
    local = torch.einsum("...ij,...kj->...ki", rot, head - ears_mid.unsqueeze(-2))
    local = local.flatten(-2)  # (..., T, 24)

    if center:
        angles = angles - angles.mean(dim=-2, keepdim=True)
        local = local - local.mean(dim=-2, keepdim=True)

    return torch.cat([angles, d_angles, centroid_vel, local], dim=-1)
