"""Obstacle tracking in the Frenet frame: N-of-M persistence plus a
constant-velocity Kalman filter per obstacle.

Each track's state is [s, d, v_s, v_d], where s is the box's rear edge (the
face seen when approaching from behind, so the most stable measurement) and
d its centre. s is kept wrapped to [0, L) and innovations use the shortest
signed difference, so tracks cross the start/finish line cleanly.

A track is reported while it was matched in at least N of the last M
scans, and dropped as soon as it can no longer reach that. A track with a
real history (at least coast_min_hits detections) is instead kept, coasting
on its prediction, until coast_time has passed since its last detection: an
opponent missed for a few scans, e.g. while its outline changes as we pull
alongside, must not vanish, while a brief phantom must not linger.

What the LiDAR sees of an obstacle changes as the car passes it (rear face,
then side face), which shifts the measured d centre by up to the box's
width, and the filter reads that as lateral velocity. So: whether a track is
moving is decided by v_s alone, a reported box takes its d extent straight
from the latest detection, and reported boxes move along s only (v_d = 0).
Opponents race along the track; a lateral velocity from changing partial
views would project an opponent we are passing sideways into our lane.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from frenet_planner.collision import ObstacleSet
from frenet_planner.frenet import s_diff

H = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0]])


@dataclass
class Track:
    id: int
    x: np.ndarray       # [s, d, v_s, v_d]
    P: np.ndarray       # 4x4 covariance
    s_len: float        # box size from the latest matched detection
    d_min: float        # d extent of the latest matched detection
    d_max: float
    t_meas: float       # time of the latest matched detection [s]
    hits: deque         # matched or not, one entry per scan, last M scans
    t: float            # time x refers to [s]
    n_hits: int = 1

    @property
    def d_width(self):
        return self.d_max - self.d_min


class ObstacleTracker:

    def __init__(self, track_length, params):
        """track_length: loop length L; params: TrackingParams."""
        self.length = track_length
        self.p = params
        self.tracks = []
        self.next_id = 0
        self.R = np.diag([params.meas_std_s ** 2, params.meas_std_d ** 2])

    # --- Kalman filter -------------------------------------------------------

    def _predict(self, tr, t):
        dt = t - tr.t
        if dt <= 0.0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # White-noise acceleration, independent in s and d
        G = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt], [dt, 0.0], [0.0, dt]])
        tr.x = F @ tr.x
        tr.x[0] = tr.x[0] % self.length
        tr.P = F @ tr.P @ F.T + self.p.accel_std ** 2 * (G @ G.T)
        tr.t = t

    def _update(self, tr, det):
        y = np.array([s_diff(det.s_min, tr.x[0], self.length),
                      0.5 * (det.d_min + det.d_max) - tr.x[1]])
        S = H @ tr.P @ H.T + self.R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ y
        tr.x[0] = tr.x[0] % self.length
        tr.P = (np.eye(4) - K @ H) @ tr.P
        tr.s_len = det.s_len
        tr.d_min, tr.d_max, tr.t_meas = det.d_min, det.d_max, tr.t

    def _spawn(self, det, t):
        P = np.diag([self.p.meas_std_s ** 2, self.p.meas_std_d ** 2,
                     self.p.init_vel_std ** 2, self.p.init_vel_std ** 2])
        x = np.array([det.s_min, 0.5 * (det.d_min + det.d_max), 0.0, 0.0])
        tr = Track(id=self.next_id, x=x, P=P, s_len=det.s_len, d_min=det.d_min, d_max=det.d_max, t_meas=t,
                   hits=deque([True], maxlen=self.p.persist_m), t=t)
        self.next_id += 1
        return tr

    # --- per scan ------------------------------------------------------------

    def step(self, detections, t):
        """Advance every track to time t and fold in one scan's detections."""
        for tr in self.tracks:
            self._predict(tr, t)

        matched_det = set()
        matched_trk = set()
        if self.tracks and detections:
            ts = np.array([tr.x[0] for tr in self.tracks])
            td = np.array([tr.x[1] for tr in self.tracks])
            zs = np.array([det.s_min for det in detections])
            zd = np.array([0.5 * (det.d_min + det.d_max) for det in detections])
            cost = np.hypot(s_diff(zs[None, :], ts[:, None], self.length), zd[None, :] - td[:, None])
            # Greedy nearest-neighbour: cheapest pairs first, each used once
            for flat in np.argsort(cost, axis=None):
                i, j = np.unravel_index(flat, cost.shape)
                if cost[i, j] > self.p.assoc_gate:
                    break
                if i in matched_trk or j in matched_det:
                    continue
                matched_trk.add(i)
                matched_det.add(j)
                self._update(self.tracks[i], detections[j])

        for i, tr in enumerate(self.tracks):
            tr.hits.append(i in matched_trk)
            tr.n_hits += i in matched_trk
        self.tracks.extend(self._spawn(det, t) for j, det in enumerate(detections) if j not in matched_det)

        self.tracks = [tr for tr in self.tracks if self._alive(tr, t)]

    def _coasting(self, tr):
        return tr.n_hits >= self.p.coast_min_hits

    def _alive(self, tr, t):
        if self._coasting(tr):
            return t - tr.t_meas <= self.p.coast_time
        n, m = self.p.persist_n, self.p.persist_m
        return sum(tr.hits) + (m - len(tr.hits)) >= n

    def confirmed(self):
        return [tr for tr in self.tracks if self._coasting(tr) or sum(tr.hits) >= self.p.persist_n]

    def is_moving(self, tr):
        return abs(tr.x[2]) >= self.p.static_speed_threshold

    def obstacle_set(self):
        """Confirmed tracks as boxes for the collision checker, at the time
        of the last step: s from the filter, d extent from the latest
        detection, velocity v_s only, and 0 for tracks with |v_s| below
        static_speed_threshold, so velocity noise does not smear a parked
        box along the track."""
        boxes = []
        for tr in self.confirmed():
            v_s = tr.x[2] if self.is_moving(tr) else 0.0
            boxes.append((tr.x[0], tr.s_len, tr.d_min, tr.d_max, v_s, 0.0))
        return ObstacleSet.from_boxes(boxes)
