import sys
src,dst=sys.argv[1],sys.argv[2]; s=open(src).read()
s=s.replace("""LOOKAHEAD_CONE_ANGLE    = math.radians(30)  # [rad] obstacles inside +/- this angle shorten the lookahead""","""LOOKAHEAD_CONE_ANGLE    = math.radians(30)  # [rad] obstacles inside +/- this angle shorten the lookahead
LOOKAHEAD_PATH_ONLY     = True  # only things in the car's own path (|y| < SPEED_PATH_HALF_WIDTH) shorten the lookahead,
                                # not bumps of a wavy side wall (they made the car weave)
TARGET_FILTER_TAU       = 0.20  # [s]  low-pass on the aim angle so the target does not jump with every wall bump (0 = off)
TARGET_FILTER_BYPASS    = 1.0   # [m]  no filtering when something is this close in the path, or while passing""",1)
old="""        closest_front = float(np.min(self.raw_ranges[np.abs(self.angles) <= LOOKAHEAD_CONE_ANGLE]))"""
new="""        if LOOKAHEAD_PATH_ONLY:
            fx = self.raw_ranges * np.cos(self.angles)
            fy = self.raw_ranges * np.sin(self.angles)
            in_path = (fx > 0) & (np.abs(fy) <= SPEED_PATH_HALF_WIDTH)
            closest_front = float(np.min(fx[in_path])) if np.any(in_path) else SPEED_RANGE_CAP
        else:
            closest_front = float(np.min(self.raw_ranges[np.abs(self.angles) <= LOOKAHEAD_CONE_ANGLE]))
        # smooth the aim angle (wall bumps make the deepest point jump left/right); react instantly when close
        if TARGET_FILTER_TAU > 0 and closest_front > TARGET_FILTER_BYPASS and not getattr(self, 'pass_side', ''):
            dt = msg.scan_time if msg.scan_time > 0 else 0.025
            alpha = dt / (TARGET_FILTER_TAU + dt)
            target_angle = self.filtered_target + alpha * (target_angle - self.filtered_target)
        self.filtered_target = target_angle"""
assert old in s; s=s.replace(old,new,1)
s=s.replace("""        self.last_steering = 0.0
""","""        self.last_steering = 0.0
        self.filtered_target = 0.0
""",1)
open(dst,'w').write(s); print('written',dst)
