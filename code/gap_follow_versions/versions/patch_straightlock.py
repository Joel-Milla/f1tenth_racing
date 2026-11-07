import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'STRAIGHT_LOCK_ENABLED' in s: print(p.split('/')[-1],'already'); continue
    s=s.replace("""LOOKAHEAD_PATH_ONLY     = True""","""STRAIGHT_LOCK_ENABLED   = True  # hold the wheels dead straight on straights instead of tiny gap-follow corrections
STRAIGHT_LOCK_DIST      = 3.0   # [m]  corridor straight ahead must be clear this far
STRAIGHT_LOCK_HALF_WIDTH= 0.30  # [m]  corridor half width (car 0.15 + 0.15 margin)
STRAIGHT_LOCK_MAX_STEER = math.radians(4)  # [rad] only lock when gap follow wants less than this
LOOKAHEAD_PATH_ONLY     = True""",1)
    old="""        steering_angle = float(np.clip(steering_angle, -MAX_STEERING_RIGHT, MAX_STEERING_LEFT))
        required_steering = steering_angle # before smoothing, used for the corner speed limit"""
    new="""        steering_angle = float(np.clip(steering_angle, -MAX_STEERING_RIGHT, MAX_STEERING_LEFT))
        self.straight_lock = False
        if STRAIGHT_LOCK_ENABLED and abs(steering_angle) < STRAIGHT_LOCK_MAX_STEER:
            sx = self.raw_ranges * np.cos(self.angles)
            sy = self.raw_ranges * np.sin(self.angles)
            in_corridor = (sx > 0) & (np.abs(sy) <= STRAIGHT_LOCK_HALF_WIDTH)
            clear = float(np.min(sx[in_corridor])) if np.any(in_corridor) else SPEED_RANGE_CAP
            if clear >= STRAIGHT_LOCK_DIST:
                steering_angle = 0.0   # straight is clear: do not scrub speed with tiny corrections
                self.straight_lock = True
        required_steering = steering_angle # before smoothing, used for the corner speed limit"""
    assert old in s,'anchor'; s=s.replace(old,new,1)
    s=s.replace("""        self.last_steering = 0.0
""","""        self.last_steering = 0.0
        self.straight_lock = False
""",1)
    s=s.replace("""f'side_block {self.side_block or "-"} | '""","""f'side_block {self.side_block or "-"} | '
                    f'{"STRAIGHT | " if self.straight_lock else ""}'""",1)
    open(p,'w').write(s); print(p.split('/')[-1],'patched')
