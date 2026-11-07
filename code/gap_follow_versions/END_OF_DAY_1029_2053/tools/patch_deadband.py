import sys
for p, deadband in [a.split('=') for a in sys.argv[1:]]:
    s = open(p).read()
    if 'STEER_SPEED_DEADBAND' not in s:
        s = s.replace("""CORNER_BRAKE_POINT      = 1.5""", """STEER_SPEED_DEADBAND    = math.radians(%s)  # [rad] steering below this = gentle/wavy bend: no corner slow-down,
                                #       only the driven arc is checked (0 = off)
CORNER_BRAKE_POINT      = 1.5""" % deadband, 1)
        s = s.replace("""        d_brake = min(self.d_ahead, d_straight)""", """        gentle = abs(steering_angle) < STEER_SPEED_DEADBAND
        # gentle bend: the wall straight ahead is not where the car is going, only the driven arc counts
        d_brake = self.d_ahead if gentle else min(self.d_ahead, d_straight)""", 1)
        s = s.replace("""        tan_steer = abs(math.tan(steering_angle))
        if tan_steer > 1e-3:""", """        tan_steer = abs(math.tan(steering_angle))
        if tan_steer > 1e-3 and not gentle:""", 1)
    open(p, 'w').write(s); print(p.split('/')[-1], 'deadband', deadband, 'refs', s.count('STEER_SPEED_DEADBAND'))
