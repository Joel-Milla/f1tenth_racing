import sys, re
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
# 1) parameters
s = s.replace("""OPP_CONFIRM_SCANS       = 5        # [-]  consecutive sightings needed before trailing (filters wall features)""",
"""OPP_CONFIRM_SCANS       = 5        # [-]  consecutive sightings needed before trailing (filters wall features)

# ---------------- Overtaking (pass instead of trailing when there is room) ----------------
PASS_ENABLED            = True
PASS_MIN_WIDTH          = 0.50     # [m]  free lateral space beside the opponent needed to pass (car 0.30 + margin)
PASS_COMMIT_TIME        = 1.0      # [s]  once a side is chosen, stick to it this long (no left/right dithering)
PASS_LOOK_BAND          = 0.40     # [m]  wall points within +/- this of the opponent's distance count for free space
PASS_MAX_RANGE          = 2.5      # [m]  only plan a pass when the opponent is closer than this""", 1)
# 2) state
s = s.replace("""        self.opp_count = 0
""", """        self.opp_count = 0
        self.opp_obs = None          # (x, y, y_left_edge, y_right_edge, angle_left_edge, angle_right_edge)
        self.pass_side = ''          # 'L' / 'R' while committed to a pass
        self.pass_until = 0.0
        self.free_left = 0.0
        self.free_right = 0.0
""", 1)
# 3) replace detect_opponent + apply_trailing with new tracking / pass / trail logic
start = s.index("    def detect_opponent(self):"); end = s.index("    def launch_active(self):")
s = s[:start] + '''    def detect_opponent(self):
        \'\'\'
        Closest car-like object ahead: (x, y, y_left_edge, y_right_edge, angle_left_edge, angle_right_edge) or None
        \'\'\'
        r = self.raw_ranges
        a = self.angles
        jumps = np.nonzero(np.abs(np.diff(r)) > OPP_JUMP)[0]  # edge between beam i and i+1
        best = None
        for k in range(len(jumps) - 1):
            i, j = jumps[k], jumps[k + 1]          # candidate object = beams i+1 .. j
            seg = r[i + 1:j + 1]
            if len(seg) == 0 or abs(a[(i + j + 1) // 2]) > OPP_FOV:
                continue
            near = float(np.median(seg))
            # both neighbours must be clearly farther (object stands out from what is behind it)
            if r[i] - near < OPP_JUMP or (j + 1 < len(r) and r[j + 1] - near < OPP_JUMP):
                continue
            width = near * (j - i) * self.latest_scan.angle_increment
            if not (OPP_MIN_WIDTH <= width <= OPP_MAX_WIDTH):
                continue
            ang = float(np.mean(a[i + 1:j + 1]))
            x, y = near * math.cos(ang), near * math.sin(ang)
            if 0.1 < x < TRAIL_DETECT_RANGE and (best is None or x < best[0]):
                a_right, a_left = float(a[i + 1]), float(a[j])   # angles grow to the left
                best = (x, y, near * math.sin(a_left), near * math.sin(a_right), a_left, a_right)
        return best

    def update_opponent(self):
        \'\'\'
        Detect and track the opponent (distance + speed estimate); runs every scan before gap selection
        \'\'\'
        now = time.monotonic()
        self.opp_obs = self.detect_opponent() if TRAIL_ENABLED else None
        if self.opp_obs is None:
            if now - self.opp_last_seen > OPP_LOST_TIME:
                self.opp_dist = None
                self.opp_speed = None
                self.opp_count = 0
            return
        x = self.opp_obs[0]
        if self.opp_dist is not None and now - self.opp_last_seen < OPP_LOST_TIME:
            dt = max(now - self.opp_last_time, 1e-3)
            raw_speed = self.speed + (x - self.opp_dist) / dt   # opponent speed = ours + gap rate
            if self.opp_speed is None:
                self.opp_speed = raw_speed  # second sighting: first real measurement
            else:
                self.opp_speed = (1 - OPP_SPEED_FILTER) * self.opp_speed + OPP_SPEED_FILTER * raw_speed
            self.opp_count += 1
        else:
            self.opp_speed = None  # first sighting: speed unknown until the next scan
            self.opp_count = 1
        self.opp_dist = x
        self.opp_last_seen = now
        self.opp_last_time = now

    def opponent_blocking(self):
        \'\'\'
        True if a confirmed, moving opponent is in our path
        \'\'\'
        if self.opp_obs is None or self.opp_speed is None or self.opp_count < OPP_CONFIRM_SCANS:
            return False
        return abs(self.opp_obs[1]) <= TRAIL_PATH_HALF_WIDTH and self.opp_speed >= OPP_STATIONARY_SPEED

    def apply_pass_mask(self):
        \'\'\'
        If a moving opponent blocks our path and one side has room, commit to that side by blanking the other
        side of the processed scan (gap follow then only sees the passing lane)
        \'\'\'
        now = time.monotonic()
        if not PASS_ENABLED:
            return
        if self.opponent_blocking() and self.opp_obs[0] < PASS_MAX_RANGE:
            x, y, y_l, y_r, a_l, a_r = self.opp_obs
            px = self.raw_ranges * np.cos(self.angles)
            py = self.raw_ranges * np.sin(self.angles)
            band = np.abs(px - x) < PASS_LOOK_BAND
            left_pts = py[band & (py > y_l + 0.02)]
            right_pts = py[band & (py < y_r - 0.02)]
            self.free_left = float(np.min(left_pts) - y_l) if left_pts.size else 2.0
            self.free_right = float(y_r - np.max(right_pts)) if right_pts.size else 2.0
            if now >= self.pass_until:
                side = 'L' if self.free_left >= self.free_right else 'R'
                if max(self.free_left, self.free_right) >= PASS_MIN_WIDTH:
                    self.pass_side = side
                    self.pass_until = now + PASS_COMMIT_TIME
                else:
                    self.pass_side = ''
        elif now >= self.pass_until:
            self.pass_side = ''
        if self.pass_side and self.opp_obs is not None:
            a_l, a_r = self.opp_obs[4], self.opp_obs[5]
            if self.pass_side == 'L':
                self.ranges[self.angles <= a_l] = 0.0   # block the opponent and everything right of it
            else:
                self.ranges[self.angles >= a_r] = 0.0   # block the opponent and everything left of it

    def apply_trailing(self, speed):
        \'\'\'
        Hold a gap behind a moving car in our path when there is no room to pass; never trail a stopped car
        \'\'\'
        self.trailing = False
        if not self.opponent_blocking() or self.pass_side:
            return speed  # nothing in the way, a stopped car (go around), or we are overtaking
        x = self.opp_obs[0]
        trail_speed = max(self.opp_speed + TRAIL_GAIN * (x - TRAIL_GAP), 0.0)
        if trail_speed < speed:
            self.trailing = True
            self.launch_done = True        # never launch into a car
            self.last_speed_cmd = trail_speed
            return trail_speed
        return speed

''' + s[end:]
# 4) call order in scan_callback: track opponent first, mask after bubble
s = s.replace("""        self.create_disparity()
        self.draw_safety_bubble()
        biggest_gap = self.find_biggest_gap()""", """        self.update_opponent()
        self.create_disparity()
        self.draw_safety_bubble()
        self.apply_pass_mask()
        biggest_gap = self.find_biggest_gap()""", 1)
# 5) log
s = s.replace("""{" TRAIL" if self.trailing else ""} | '""", """{" TRAIL" if self.trailing else ""}{" PASS-" + self.pass_side if self.pass_side else ""} | '""", 1)
open(dst, 'w').write(s); print('written', dst, 'PASS refs', s.count('PASS_'))
