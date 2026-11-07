import sys
for p in sys.argv[1:]:
    s = open(p).read()
    if 'TRAIL_ENABLED' in s: print(p.split('/')[-1], 'already patched'); continue
    s = s.replace("""# ---------------- Deadman switch (hold R1 to drive) ----------------""", """# ---------------- Opponent trailing (head-to-head) ----------------
# A car ahead = short object (OPP_MIN..MAX_WIDTH) with a range jump > OPP_JUMP on BOTH edges (walls are continuous).
# If it is in our path and moving, cap our speed to hold TRAIL_GAP behind it. A stopped car (< OPP_STATIONARY_SPEED)
# is not trailed: gap follow drives around it like the qualifying obstacle box.
TRAIL_ENABLED           = True
TRAIL_DETECT_RANGE      = 3.0      # [m]  only look for opponents this far ahead
TRAIL_PATH_HALF_WIDTH   = 0.35     # [m]  opponent counts as "in our path" if its centre is within this lateral offset
TRAIL_GAP               = 0.8      # [m]  distance to hold behind the opponent (front of lidar to its back)
TRAIL_GAIN              = 1.5      # [1/s] speed correction per metre of gap error
OPP_MIN_WIDTH           = 0.08     # [m]
OPP_MAX_WIDTH           = 0.55     # [m]
OPP_JUMP                = 0.30     # [m]  range jump needed on both edges of the object
OPP_FOV                 = math.radians(70)  # [rad] search +/- this angle
OPP_STATIONARY_SPEED    = 0.5      # [m/s] slower than this = stopped car / obstacle -> no trailing, go around
OPP_SPEED_FILTER        = 0.15     # [-]  low-pass weight per scan for the opponent speed estimate (~0.3 s)
OPP_LOST_TIME           = 0.3      # [s]  forget the opponent if not seen for this long

# ---------------- Deadman switch (hold R1 to drive) ----------------""", 1)
    s = s.replace("""        # launch boost: runs once, starting the first time the car is allowed to drive""", """        # opponent tracking for trailing
        self.opp_dist = None
        self.opp_speed = None
        self.opp_last_seen = 0.0
        self.opp_last_time = 0.0
        self.trailing = False

        # launch boost: runs once, starting the first time the car is allowed to drive""", 1)
    s = s.replace("""        speed = self.rate_limit_speed(speed)
""", """        speed = self.rate_limit_speed(speed)
        speed = self.apply_trailing(speed)
""", 1)
    s = s.replace("""                    f'side_block {self.side_block or "-"} | '""", """                    f'side_block {self.side_block or "-"} | '
                    f'opp {"-" if self.opp_dist is None else f"{self.opp_dist:.2f}m@{self.opp_speed:.1f}"}{" TRAIL" if self.trailing else ""} | '""", 1)
    s = s.replace("""    def launch_active(self):""", """    def detect_opponent(self):
        '''
        Returns (distance_ahead, lateral_offset) of the closest car-like object ahead, or None
        '''
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
                best = (x, y)
        return best

    def apply_trailing(self, speed):
        '''
        Hold a gap behind a moving car in our path; never trail a stopped car (drive around it instead)
        '''
        self.trailing = False
        if not TRAIL_ENABLED:
            return speed
        now = time.monotonic()
        opp = self.detect_opponent()
        if opp is None:
            if now - self.opp_last_seen > OPP_LOST_TIME:
                self.opp_dist = None
                self.opp_speed = None
            return speed
        x, y = opp
        if self.opp_dist is not None and now - self.opp_last_seen < OPP_LOST_TIME:
            dt = max(now - self.opp_last_time, 1e-3)
            raw_speed = self.speed + (x - self.opp_dist) / dt   # opponent speed = ours + gap rate
            self.opp_speed = (1 - OPP_SPEED_FILTER) * self.opp_speed + OPP_SPEED_FILTER * raw_speed
        else:
            self.opp_speed = self.speed  # first sighting: assume same speed until measured
        self.opp_dist = x
        self.opp_last_seen = now
        self.opp_last_time = now
        if abs(y) > TRAIL_PATH_HALF_WIDTH or self.opp_speed < OPP_STATIONARY_SPEED:
            return speed  # beside our path, or stopped: let gap follow go around it
        trail_speed = max(self.opp_speed + TRAIL_GAIN * (x - TRAIL_GAP), 0.0)
        if trail_speed < speed:
            self.trailing = True
            self.launch_done = True        # never launch into a car
            self.last_speed_cmd = trail_speed
            return trail_speed
        return speed

    def launch_active(self):""", 1)
    open(p, 'w').write(s); print(p.split('/')[-1], 'patched:', s.count('TRAIL_'), 'refs')
