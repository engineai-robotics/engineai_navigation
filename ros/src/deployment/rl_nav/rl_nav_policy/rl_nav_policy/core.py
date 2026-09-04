"""Numerical utilities for the PM01 lidar navigation policy."""

import math
from typing import Optional, Tuple

import numpy as np


NUM_CHANNELS = 16
NUM_AZIMUTHS = 41
NUM_POINTS = NUM_CHANNELS * NUM_AZIMUTHS
OBSERVATION_DIM = 3 + 3 + 3 + 3 + 4 + NUM_POINTS * 3
LSTM_SIZE = 512


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return a normalized xyzw quaternion with a deterministic sign."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("quaternion must be a finite, non-zero xyzw vector")
    result = quaternion / norm
    if result[3] < 0.0:
        result = -result
    return result


def quaternion_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of an xyzw quaternion."""
    x, y, z, w = quaternion
    return np.asarray((-x, -y, -z, w), dtype=np.float64)


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton-multiply two xyzw quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion into a 3x3 active rotation matrix."""
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def yaw_from_xyzw(quaternion: np.ndarray) -> float:
    """Extract yaw from an xyzw quaternion."""
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def lidar_observation(
    endpoints_body: np.ndarray,
    quaternion_xyzw: np.ndarray,
    ray_origin: np.ndarray,
    max_distance: float,
) -> Tuple[np.ndarray, int]:
    """Convert organized body-frame endpoints to normalized yaw-frame XYZ."""
    endpoints = np.asarray(endpoints_body, dtype=np.float64)
    if endpoints.shape != (NUM_POINTS, 3):
        raise ValueError(f"expected lidar shape ({NUM_POINTS}, 3)")
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive")
    origin = np.asarray(ray_origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("ray_origin must be a finite 3-vector")

    valid = np.all(np.isfinite(endpoints), axis=1)
    valid &= np.sum(endpoints * endpoints, axis=1) > 1.0e-24
    relative = np.zeros_like(endpoints)
    relative[valid] = endpoints[valid] - origin

    rotation = rotation_matrix_xyzw(quaternion_xyzw)
    yaw = yaw_from_xyzw(quaternion_xyzw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    inverse_heading = np.asarray(
        (
            (cos_yaw, sin_yaw, 0.0),
            (-sin_yaw, cos_yaw, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    points_yaw = (inverse_heading @ rotation @ relative.T).T
    points_yaw[~valid] = 0.0
    result = (points_yaw / max_distance).astype(np.float32).reshape(-1)
    return result, int(np.count_nonzero(valid))


def projected_gravity(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Project the world down vector into the robot body frame."""
    rotation = rotation_matrix_xyzw(quaternion_xyzw)
    return (rotation.T @ np.asarray((0.0, 0.0, -1.0))).astype(np.float32)


def goal_observation(
    position_world: np.ndarray,
    quaternion_xyzw: np.ndarray,
    goal_world: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return body-frame goal direction, log range, and planar range."""
    position = np.asarray(position_world, dtype=np.float64)
    goal = np.asarray(goal_world, dtype=np.float64)
    if position.shape != (3,) or goal.shape != (3,):
        raise ValueError("position and goal must be 3-vectors")
    goal_body = rotation_matrix_xyzw(quaternion_xyzw).T @ (goal - position)
    distance = float(np.linalg.norm(goal_body))
    direction = goal_body / max(distance, 1.0e-6)
    observation = np.concatenate(
        (direction, np.asarray((math.log(distance + 1.0),)))
    ).astype(np.float32)
    planar_distance = float(np.linalg.norm(goal[:2] - position[:2]))
    return observation, planar_distance


class PoseVelocityEstimator:
    """Estimate body velocities from timestamped world-frame poses."""

    def __init__(self, filter_alpha: float = 0.8, max_dt: float = 0.5):
        if not 0.0 <= filter_alpha < 1.0:
            raise ValueError("filter_alpha must be in [0, 1)")
        if not np.isfinite(max_dt) or max_dt <= 0.0:
            raise ValueError("max_dt must be positive")
        self.filter_alpha = float(filter_alpha)
        self.max_dt = float(max_dt)
        self.previous: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        self.linear_body = np.zeros(3, dtype=np.float32)
        self.angular_body = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        """Clear pose history and filtered velocities."""
        self.previous = None
        self.linear_body.fill(0.0)
        self.angular_body.fill(0.0)

    def update(
        self,
        timestamp: float,
        position_world: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Update from a pose and return linear/angular body velocities."""
        position = np.asarray(position_world, dtype=np.float64)
        quaternion = normalize_quaternion_xyzw(quaternion_xyzw)
        if (
            not np.isfinite(timestamp)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
        ):
            raise ValueError("pose and timestamp must be finite")

        if self.previous is None:
            self.previous = (float(timestamp), position.copy(), quaternion)
            return self.linear_body.copy(), self.angular_body.copy(), False

        previous_time, previous_position, previous_quaternion = self.previous
        dt = float(timestamp - previous_time)
        self.previous = (float(timestamp), position.copy(), quaternion)
        if dt <= 1.0e-5 or dt > self.max_dt:
            self.linear_body.fill(0.0)
            self.angular_body.fill(0.0)
            return self.linear_body.copy(), self.angular_body.copy(), False

        linear_world = (position - previous_position) / dt
        linear_body = rotation_matrix_xyzw(quaternion).T @ linear_world

        delta = quaternion_multiply_xyzw(
            quaternion_conjugate_xyzw(previous_quaternion), quaternion
        )
        delta = normalize_quaternion_xyzw(delta)
        vector = delta[:3]
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm < 1.0e-9:
            angular_body = np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * math.atan2(vector_norm, delta[3])
            angular_body = vector / vector_norm * angle / dt

        blend = 1.0 - self.filter_alpha
        self.linear_body = (
            self.filter_alpha * self.linear_body + blend * linear_body
        ).astype(np.float32)
        self.angular_body = (
            self.filter_alpha * self.angular_body + blend * angular_body
        ).astype(np.float32)
        return self.linear_body.copy(), self.angular_body.copy(), True


def build_observation(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    gravity: np.ndarray,
    last_raw_action: np.ndarray,
    goal: np.ndarray,
    lidar: np.ndarray,
) -> np.ndarray:
    """Assemble the exact actor observation ordering used during training."""
    observation = np.concatenate(
        (
            np.asarray(linear_velocity, dtype=np.float32),
            np.asarray(angular_velocity, dtype=np.float32),
            np.asarray(gravity, dtype=np.float32),
            np.asarray(last_raw_action, dtype=np.float32),
            np.asarray(goal, dtype=np.float32),
            np.asarray(lidar, dtype=np.float32),
        )
    )
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"expected {OBSERVATION_DIM} observation values, "
            f"got {observation.shape}"
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError("policy observation contains non-finite values")
    return observation[None, :]
