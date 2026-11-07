import sys
for p in sys.argv[1:]:
    s = open(p).read()
    if 'DEADMAN_BUTTON' in s:
        print(p.split('/')[-1], 'already has deadman'); continue
    s = s.replace("from sensor_msgs.msg import LaserScan\n", "from sensor_msgs.msg import LaserScan, Joy\n", 1)
    s = s.replace("""ODOM_TIMEOUT            = 0.2      # [s] odom older than this -> use the VESC speed instead
""", """ODOM_TIMEOUT            = 0.2      # [s] odom older than this -> use the VESC speed instead

# ---------------- Deadman switch (hold R1 to drive) ----------------
DEADMAN_ENABLED         = True     # False = drive as soon as the node starts (old behaviour)
DEADMAN_BUTTON          = 5        # joy button index: 5 = R1 / RB (same as joy_teleop autonomous_control)
JOY_TOPIC               = 'joy'
DEADMAN_TIMEOUT         = 0.3      # [s] no /joy message for this long (controller lost) -> stop
DEADMAN_STOP_TIME       = 0.5      # [s] after release, keep sending stop commands this long, then go quiet
                                   #     so manual driving (L1) through the mux works again
""", 1)
    s = s.replace("""        self.publisher_vel = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, QOS_DEPTH)
""", """        self.publisher_vel = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, QOS_DEPTH)

        # deadman switch: only drive while R1 is held
        self.deadman_pressed = False
        self.last_joy_time = 0.0
        self.deadman_release_time = 0.0
        self.subscription_joy = self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, QOS_DEPTH)
""", 1)
    s = s.replace("""    def vesc_state_callback(self, msg):""", """    def joy_callback(self, msg : Joy):
        '''
        Tracks the deadman button (R1)
        '''
        pressed = len(msg.buttons) > DEADMAN_BUTTON and msg.buttons[DEADMAN_BUTTON] == 1
        if self.deadman_pressed and not pressed:
            self.deadman_release_time = time.monotonic()
        self.deadman_pressed = pressed
        self.last_joy_time = time.monotonic()

    def deadman_active(self):
        '''
        True while R1 is held and the controller is still sending messages
        '''
        if not DEADMAN_ENABLED:
            return True
        now = time.monotonic()
        if self.deadman_pressed and now - self.last_joy_time > DEADMAN_TIMEOUT:
            # controller went quiet while held -> treat as released
            self.deadman_pressed = False
            self.deadman_release_time = now
        return self.deadman_pressed

    def vesc_state_callback(self, msg):""", 1)
    s = s.replace("""        self.publish_velocity(self.boost_speed_command(speed), steering_angle)
""", """        if self.deadman_active():
            self.publish_velocity(self.boost_speed_command(speed), steering_angle)
        else:
            # not held: reset the speed ramp so it starts gently next time, send stop briefly, then go quiet
            self.last_speed_cmd = 0.0
            if time.monotonic() - self.deadman_release_time < DEADMAN_STOP_TIME:
                self.publish_velocity(0.0, 0.0)
""", 1)
    open(p, 'w').write(s)
    print(p.split('/')[-1], 'patched:', s.count('DEADMAN_BUTTON'), 'refs')
