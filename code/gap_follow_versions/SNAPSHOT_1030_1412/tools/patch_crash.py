import sys
for p in sys.argv[1:]:
    s = open(p).read(); n = 0
    old = """        v_clear = math.sqrt(CORNER_SPEED**2 + 2.0 * BRAKE_DECEL * max(d_brake - CORNER_BRAKE_POINT, 0.0))"""
    new = """        v_clear = math.sqrt(CORNER_SPEED**2 + 2.0 * BRAKE_DECEL * max(d_brake - CORNER_BRAKE_POINT, 0.0))
        # inside the brake point, something close ahead must still slow the car toward a stop
        v_clear = min(v_clear, math.sqrt(2.0 * BRAKE_DECEL * max(d_brake - STOP_MARGIN, 0.0)) + CORNER_SPEED * 0.5)"""
    if old in s and 'something close ahead must still slow' not in s: s = s.replace(old, new, 1); n += 1
    old2 = """        def held(side):
            return self.side_hold[side][0] > 0 and self.side_hold[side][1] > 0
"""
    new2 = """        def held(side):
            return self.side_hold[side][0] > 0 and self.side_hold[side][1] > 0

        # walls close on both sides (narrow section): holding straight never helps, so do not block
        if held('L') and held('R'):
            return steering_angle
"""
    if old2 in s and 'narrow section): holding straight' not in s: s = s.replace(old2, new2, 1); n += 1
    s2 = s.replace("SIDE_CLEARANCE          = 0.30 ", "SIDE_CLEARANCE          = 0.25 ")
    n += s2 != s; s = s2
    open(p, 'w').write(s); print(p.split('/')[-1], 'changes:', n)
