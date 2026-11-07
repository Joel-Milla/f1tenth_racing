"""Localization health, the scan-to-map mismatch and the follow-the-gap adapter."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import CONFIG_YAML
from frenet_planner.health import PF3X3, ROS6X6, GapFollowAdapter, LocalizationHealth, pose_std
from frenet_planner.params import Params
from test_detection import ANGLE_INC, ANGLE_MIN, car_pose, detector, params, scan  # noqa: F401 (fixtures)
from test_track_bounds import circle_ref  # noqa: F401 (fixture)


@pytest.fixture
def hp():
    return Params.from_yaml(CONFIG_YAML).health


def test_pose_std_layouts():
    pf = np.zeros(36)
    pf[[0, 4, 8]] = (0.04, 0.05, 0.01)          # particle_filter packs a 3x3 first
    assert pose_std(pf, PF3X3) == pytest.approx((0.3, 0.1))
    ros = np.zeros(36)
    ros[[0, 7, 35]] = (0.04, 0.05, 0.01)
    assert pose_std(ros, ROS6X6) == pytest.approx((0.3, 0.1))
    assert pose_std(np.zeros(36), PF3X3) is None  # the sim: no covariance


def test_hysteresis(hp):
    h = LocalizationHealth(hp)
    bad, good = hp.max_mismatch_fraction + 0.1, 0.0
    for _ in range(hp.unhealthy_after - 1):
        assert h.update(bad, 1000)
    assert not h.update(bad, 1000)
    for _ in range(hp.healthy_after - 1):
        assert not h.update(good, 1000)
    assert h.update(good, 1000)


def test_covariance_alone_makes_it_unhealthy(hp):
    h = LocalizationHealth(hp)
    h.set_pose_std((hp.max_position_std + 0.1, 0.0))
    for _ in range(hp.unhealthy_after):
        h.update(0.0, 1000)
    assert not h.healthy and 'position std' in h.reason


def test_too_few_points_gives_no_verdict(hp):
    h = LocalizationHealth(hp)
    for _ in range(3 * hp.unhealthy_after):
        h.update(1.0, hp.min_scan_points - 1)
    assert h.healthy


def test_mismatch_is_low_with_the_true_pose_and_high_with_a_wrong_one(detector, hp):  # noqa: F811
    pose = car_pose(0.0)
    ranges = scan(pose, [])
    good = detector.detect(ranges, ANGLE_MIN, ANGLE_INC, 0.0, pose, 0.0, hp.mismatch_dist)
    off = (pose[0] - 0.5, pose[1], pose[2] + 0.2)   # half a metre and 11 deg off
    bad = detector.detect(ranges, ANGLE_MIN, ANGLE_INC, 0.0, off, 0.0, hp.mismatch_dist)
    assert good.mismatch < 0.05
    assert bad.mismatch > hp.max_mismatch_fraction


class FakeReactive:
    """Stands in for the team's gap_follow reactive_node module: same class
    name and the attributes its lidar_callback relies on."""

    class ReactiveFollowGap:
        def __init__(self):
            raise AssertionError('the adapter must not run __init__ (it starts ROS)')

        def lidar_callback(self, data):
            i = int(np.argmax(np.minimum(np.asarray(data.ranges), self.range_max)))
            msg = SimpleNamespace(drive=SimpleNamespace(
                steering_angle=data.angle_min + i * data.angle_increment, speed=self.bubble_radius))
            self.drivexpub.publish(msg)


def test_adapter_runs_the_callback_and_captures_its_command(hp):
    gp = dataclasses.replace(Params.from_yaml(CONFIG_YAML).gap_follow, bubble_radius=0.7)
    a = GapFollowAdapter(FakeReactive, gp)
    ranges = np.full(11, 1.0)
    ranges[8] = 5.0
    steer, speed = a.drive(ranges, -0.5, 0.1)
    assert steer == pytest.approx(-0.5 + 0.8)
    assert speed == pytest.approx(0.7)
    assert a.ftg.fov_half == pytest.approx(np.radians(gp.fov_half_deg))
