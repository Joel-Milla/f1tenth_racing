import sys
for p in sys.argv[1:]:
    s = open(p).read()
    if 'LAUNCH_ENABLED' in s: print(p.split('/')[-1], 'already patched'); continue
    s = s.replace("""# ---------------- Deadman switch (hold R1 to drive) ----------------""", """# ---------------- Launch boost (race start) ----------------
LAUNCH_ENABLED          = True     # first LAUNCH_TIME after R1 is first pressed: full speed, steering from gap follow
LAUNCH_TIME             = 1.0      # [s]
LAUNCH_SPEED            = 5.0      # [m/s] command during launch (VESC max ~5.0; the car reaches ~2.5-3.5 in 1 s)
LAUNCH_ABORT_DIST       = 1.0      # [m] skip the launch if anything is this close straight ahead (0 = never skip)

# ---------------- Deadman switch (hold R1 to drive) ----------------""", 1)
    s = s.replace("""        self.subscription_joy = self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, QOS_DEPTH)
""", """        self.subscription_joy = self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, QOS_DEPTH)

        # launch boost: runs once, starting the first time the car is allowed to drive
        self.launch_start = None
        self.launch_done = not LAUNCH_ENABLED
""", 1)
    s = s.replace("""        if self.deadman_active():
            self.publish_velocity(self.boost_speed_command(speed), steering_angle)""", """        if self.deadman_active():
            if self.launch_active():
                self.publish_velocity(LAUNCH_SPEED, steering_angle)
            else:
                self.publish_velocity(self.boost_speed_command(speed), steering_angle)""", 1)
    s = s.replace("""    def deadman_active(self):""", """    def launch_active(self):
        '''
        True during the first LAUNCH_TIME seconds of driving (once per node run)
        '''
        if self.launch_done:
            return False
        now = time.monotonic()
        if self.launch_start is None:
            x = self.raw_ranges * np.cos(self.angles)
            y = self.raw_ranges * np.sin(self.angles)
            ahead = (x > 0) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
            d_ahead = float(np.min(x[ahead])) if np.any(ahead) else SPEED_RANGE_CAP
            if d_ahead < LAUNCH_ABORT_DIST:
                self.get_logger().info(f'launch skipped: obstacle {d_ahead:.2f} m ahead')
                self.launch_done = True
                return False
            self.launch_start = now
            self.get_logger().info('launch!')
        if now - self.launch_start < LAUNCH_TIME:
            return True
        # hand over to normal control from the current speed so it does not drop suddenly
        self.launch_done = True
        self.last_speed_cmd = max(self.speed, 0.0)
        self.last_cmd_time = now
        return False

    def deadman_active(self):""", 1)
    open(p, 'w').write(s); print(p.split('/')[-1], 'patched:', s.count('LAUNCH_'), 'refs')
