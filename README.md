# Team Roku – F1TENTH backup (CMU 16-665 race, Oct/Nov 2026)

Backup of the car before it was wiped. Seed 1 in qualifying (7.511 s best lap).

## Race code (what actually raced)

`code/f1tenth_ws_src/gap_follow/gap_follow/gap_follow_params.py` is the live file the car ran in the final rounds.
It is identical to `code/gap_follow_versions/RACE_GOOD_track3.py`.

Features (all switchable by the parameter block at the top of the file):
- Reactive gap follow (disparity extender + safety bubble), vectorised with numpy (~1 ms per scan)
- Pure-pursuit steering toward the deepest open region, steering rate limit + smoothing
- Weaving fix: only obstacles in the car's own path shorten the lookahead (`LOOKAHEAD_PATH_ONLY`)
- Straight lock: wheels held straight on clear straights (`STRAIGHT_LOCK_ENABLED`)
- Continuous speed: brake-for-the-corner (`CORNER_SPEED`, `CORNER_BRAKE_POINT`), cornering limit (`LATERAL_ACCEL`),
  brake/accel boost to over-command the VESC, gentle-bend deadband, near-wall slowdown
- Launch boost: 5.0 m/s for up to 3 s after the wheels start moving, ends early before a corner
- Trailing: detects a moving car ahead and holds 0.8 m behind it (no passing in the race version)
- R1 deadman (hold to drive), VESC speed fallback if `/odom` stops

Run it (bringup running, ROS_DOMAIN_ID=2):
```
python3 gap_follow_params.py
```

## Layout
- `code/f1tenth_ws_src/` – the whole ROS 2 workspace `src/` (all packages) as it was on the car.
  - `code/UNCOMMITTED_CHANGES_in_src_repo.txt`, `code/uncommitted_*.diff` – what was not committed to the
    original repo (github.com/aswarthm/f1tenth-robot-mobility). Includes `vesc.yaml` `servo_min 0.15 -> 0.10`.
  - `code/*_history.bundle` – full git history of the original repos. Restore with
    `git clone code/src_repo_history.bundle restored_src` (same for f1tenth_system and grok_nav).
- `code/gap_follow_versions/` – every tuned version (FAST*, RACE_*, *_today*, PRE_* backups),
  `SPEED_CHEATSHEET.txt` (what every parameter does and the history of changes), snapshots.
- `code/gap_follow_versions/tools/` – recorders, analysis scripts, grip test, and the simulators:
  `sim_map.py` (closed-loop sim on a track map), `sim_pass.py`, `sim_wavy.py`, `sim_corners.py`.
  Example: `MAP=maps/track3_clean.pgm START=124,140,16 python3 sim_map.py "name|path/to/version.py"`
- `code/notes/` – tuning notes (measured limits, decisions).
- `maps/` – track maps (`track2_clean.pgm/.yaml`, `track3_clean.pgm`, 0.05 m/px) and track photos.
- `data/` (separate archive, not in git) – run recordings (`runs/*.pkl`: lidar scan + drive command +
  speed + IMU + R1 per scan), node logs, and the race videos (`ego.mp4`, `top_down.mp4`).

## Measured car limits
- Grip: hard floor ~3.3 m/s², carpet sliding starts ~5.5–6 m/s², peak ~7 m/s² (IMU circle test)
- Real braking ~1.3 m/s² (VESC brake current limit); VESC reaches ~80 % of the commanded speed
- Real wheelbase ~0.25 m (code uses 0.33, tuned around it); ~0.35 s from first command to wheels moving
- Race laps on track3: ~7.8 s real (simulator says ~7.0 s; it has no tyre sliding)
