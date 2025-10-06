import numpy as np

from transforms3d.euler import axangle2euler


def convert_axangle_to_rpy(axangle):
    action_rotation_delta = axangle.astype(np.float64)
    action_rotation_angle = np.linalg.norm(action_rotation_delta)
    action_rotation_ax = (
        action_rotation_delta / action_rotation_angle
        if action_rotation_angle > 1e-6
        else np.array([0.0, 1.0, 0.0])
    )
    roll, pitch, yaw = axangle2euler(action_rotation_ax, action_rotation_angle)
    return np.array([roll, pitch, yaw], dtype=axangle.dtype)
