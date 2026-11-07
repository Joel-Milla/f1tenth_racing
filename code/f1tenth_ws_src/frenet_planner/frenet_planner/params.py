"""Typed planner parameters. Loaded from config/frenet_planner.yaml (or from the
ROS parameter server by the node). Every tunable lives in the YAML; the
dataclasses only give them names and types, so there are no defaults here."""
import math
from dataclasses import dataclass, fields, is_dataclass

import yaml

NODE_NAME = 'frenet_planner'


@dataclass
class RacelineParams:
    file: str
    smoothing: float
    ds: float
    search_back: float
    search_fwd: float
    reset_dist: float


@dataclass
class MapParams:
    yaml_file: str
    unknown_is_occupied: bool


@dataclass
class TrackParams:
    ds: float
    max_range: float
    min_filter: float


@dataclass
class VehicleParams:
    wheelbase: float
    max_steer: float
    width: float
    length: float

    @property
    def kappa_max(self):
        """Largest path curvature the steering can produce (kinematic bicycle)."""
        return math.tan(self.max_steer) / self.wheelbase


@dataclass
class CollisionParams:
    n_circles: int
    footprint_center_offset: float
    obstacle_margin: float


@dataclass
class DetectionParams:
    laser_offset_x: float
    max_range: float
    wall_margin: float
    track_bound_tolerance: float
    frenet_window_back: float
    frenet_window_fwd: float
    breakpoint_lambda: float
    range_sigma: float
    max_beam_gap: int
    min_cluster_points: int
    max_cluster_size: float
    merge_gap: float
    min_s_extent: float


@dataclass
class TrackingParams:
    persist_n: int
    persist_m: int
    assoc_gate: float
    meas_std_s: float
    meas_std_d: float
    accel_std: float
    init_vel_std: float
    static_speed_threshold: float
    coast_time: float
    coast_min_hits: int


@dataclass
class SpeedParams:
    v_max: float
    a_lat_max: float
    a_accel_max: float
    a_decel_max: float
    global_scale: float


@dataclass
class NodeParams:
    """Used only by the ROS wrapper."""
    pose_topic: str
    scan_topic: str
    drive_topic: str
    start_armed: bool
    frame_id: str
    viz_rate: float
    static_viz_period: float


@dataclass
class CostWeights:
    offset: float
    jerk: float
    curvature: float
    obstacle: float
    switch: float
    length: float


@dataclass
class PlannerParams:
    rate: float
    ds: float
    path_length: float
    n_offsets: int
    horizon_times: list
    horizon_min: float
    horizon_max: float
    v_plan_min: float
    bound_margin: float
    bounds_grace: float
    kappa_ref_slack: float
    obstacle_behind: float
    state_reset_dist: float
    v_avoid: float
    v_fallback: float
    stop_margin: float
    weights: CostWeights


@dataclass
class BehaviorParams:
    clear_dist: float
    opponent_range: float
    trail_gap: float
    trail_min_gap: float
    trail_gain: float
    drop_back_speed: float
    overtake_margin: float
    overtake_keep_margin: float
    overtake_cooldown: float
    overtake_min_advantage: float
    pass_offset: float
    pass_length: float
    overtake_obstacle_weight: float


@dataclass
class HealthParams:
    fallback_enabled: bool
    covariance_layout: str
    max_position_std: float
    max_yaw_std: float
    mismatch_dist: float
    max_mismatch_fraction: float
    min_scan_points: int
    unhealthy_after: int
    healthy_after: int
    handback_max_heading: float


@dataclass
class GapFollowParams:
    """Attributes of the team's ReactiveFollowGap (gap_follow/reactive_node.py)."""
    range_max: float
    bubble_radius: float
    corner_safe_dist: float
    fov_half_deg: float


@dataclass
class ControlParams:
    rate: float
    lookahead_gain: float
    lookahead_base: float
    lookahead_min: float
    lookahead_max: float


@dataclass
class TestObstacleParams:
    """RViz-clicked boxes the planner avoids (the simulator does not see them)."""
    size: float
    max_count: int


@dataclass
class Params:
    node: NodeParams
    raceline: RacelineParams
    map: MapParams
    track: TrackParams
    vehicle: VehicleParams
    collision: CollisionParams
    detection: DetectionParams
    tracking: TrackingParams
    speed: SpeedParams
    planner: PlannerParams
    behavior: BehaviorParams
    control: ControlParams
    health: HealthParams
    gap_follow: GapFollowParams
    test_obstacles: TestObstacleParams

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d, '')

    @classmethod
    def from_yaml(cls, path, node_name=NODE_NAME):
        with open(path) as f:
            doc = yaml.safe_load(f)
        return cls.from_dict(doc[node_name]['ros__parameters'])


def _build(cls, d, prefix):
    """Recursively build a dataclass from a nested dict. Missing and unknown
    keys are errors, so a typo in the YAML fails loudly at startup."""
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    missing = set(names) - set(d)
    if unknown or missing:
        raise ValueError(f'{cls.__name__} at "{prefix or "<root>"}": '
                         f'unknown keys {sorted(unknown)}, missing keys {sorted(missing)}')
    kwargs = {}
    for name, f in names.items():
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, d[name], f'{prefix}{name}.')
        else:
            kwargs[name] = f.type(d[name])
    return cls(**kwargs)
