---
name: gap-follow-race-tuning
description: "State of the gap_follow_params.py race tuning (Oct 2026): race version, fallbacks, measured car limits, backups/tools location"
metadata:
  node_type: memory
  type: project
  originSessionId: fee9f4c2-6d52-40c9-8f40-96439f544a19
  modified: 2026-10-30T18:12:35.199Z
---

Tuning src/gap_follow/gap_follow/gap_follow_params.py for an F1TENTH race on carpet (narrow ~1.1 m wavy track). User wants only that file edited (servo_min in vesc.yaml changed to 0.10 with permission).

2026-10-30 14:12 (LATEST, user: "really good"): live = ~/gap_follow_backups/RACE_GOOD_track3.py (= RACE_SAFE_zone_boost3: trailing w/ trail-zone fix + no trailing at standstill, 3 s launch w/ early hand-over, weaving fix, straight lock, no passing). Full snapshot SNAPSHOT_1030_1412. Today's track map: ~/f1tenth_ws/track3_clean.pgm (no yaml; sim with MAP=... START=124,140,16, anticlockwise from bottom straight).
2026-10-30 (earlier): race plan is trail-only, no passing. Live = RACE_SAFE_today.py; set = FAST6_today, FAST6_trail_today (=RACE_FAST_today), RACE_SAFE_today — all with weaving fix, PASS_ENABLED False, TRAIL_PATH_HALF_WIDTH 0.55.
2026-10-30: after video review, live = RACE_SAFE_v2.py (LOOKAHEAD_PATH_ONLY fix for weaving + TARGET_DEEP_FRACTION 0.93); pace option RACE_FAST_v2.py. Real track map: ~/f1tenth_ws/track2_clean.pgm/.yaml; closed-loop sim on it: ~/gap_follow_backups/tools/sim_map.py (matched real ~8 s laps). Aim filters and centre-of-gap aiming were worse in sim.
Race-day strategy (2026-10-29, Roku = seed 1, 7.511 s quali; races first-to-10-laps, possibly side-by-side start): live = RACE_SAFE_pass.py (1.5 s launch boost, FAST4 speeds, opponent trail/pass/follow logic). Faster variant with same logic: FAST6_pass.py. Pass logic validated in a 2D ray-cast sim (tools/sim_pass.py) and 0 false triggers on solo recordings (tools/trail_fp.py). Open issue: ~3 s delay before wheels move after first command — user should time it on the stand.
End of session 2026-10-29 (before pass work): live file = FAST6 + 1.5 s launch (backup LAST_RUN_FAST6.py; avg 3.0 m/s, lots of wall contact). FAST7 crashed. FAST5 best measured lap 8.65 s. FAST4-6 add launch boost, last-curve arc-only braking, less inflation.
Earlier race version (2026-10-29): ~/gap_follow_backups/RACE_FAST3_good.py (= FAST3_carpet.py, live file): ~8.95 s lap, 2.67 m/s avg, calm steering. Fallbacks in order: FAST2_carpet.py -> SAFE_carpet.py -> SAFE_floor.py. All have: R1 deadman (button 5), VESC speed fallback if /odom is down, SIDE_CHECK off (it blocked turns on the narrow track), slow-down near walls, CORNER_SPEED/CORNER_BRAKE_POINT braking, STEER_SPEED_DEADBAND (10 deg only in FAST3).

**Why:** grip measured by IMU circle test: hard floor ~3.3 m/s^2, carpet understeer starts ~5.5-6, peak ~7. Real braking ~1.3 m/s^2 (VESC brake current). Car's real wheelbase ~0.25 m (file uses 0.33, left as is). Car sits ~3 s after node start before moving.

**How to apply:** user says start/stop; Claude runs `timeout -s INT 60 python <live file>` plus ~/gap_follow_backups/tools/rec3.py and analyze.py/lap.py in ~/gap_follow_backups/runs. Never use pkill/pgrep -f with a pattern (even a [b]racket regex) in a Bash call whose own text contains the matched path — it kills Claude's own shell. Stop processes by the saved pid (~/gap_follow_backups/runs/node_pid.txt) in a separate call. Continuous runs: nohup node + tools/rec4.py (saves every 10 s, SIGTERM to stop). Cheat sheet: ~/gap_follow_backups/SPEED_CHEATSHEET.txt. ROS_DOMAIN_ID=2.
