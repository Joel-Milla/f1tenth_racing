import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'TRAIL_OFFSET_MIN_DIST' in s: print(p.split('/')[-1],'already'); continue
    s=s.replace("""TRAIL_GAP               = 0.8""","""TRAIL_OFFSET_MIN_DIST   = 0.8      # [m]  an object offset more than 0.35 m only counts if it is at least this far ahead
TRAIL_OFFSET_MAX_RATIO  = 0.6      # [-]  ...and in front, not beside: |y| < this * x  (~30 deg); stops wall bumps next to
                                   #      the car from triggering trailing
TRAIL_GAP               = 0.8""",1)
    old="""        return abs(self.opp_obs[1]) <= TRAIL_PATH_HALF_WIDTH and self.opp_speed >= OPP_STATIONARY_SPEED"""
    new="""        x, y = self.opp_obs[0], abs(self.opp_obs[1])
        in_path = y <= 0.35 or (y <= TRAIL_PATH_HALF_WIDTH and x >= TRAIL_OFFSET_MIN_DIST and y < TRAIL_OFFSET_MAX_RATIO * x)
        return in_path and self.opp_speed >= OPP_STATIONARY_SPEED"""
    assert old in s,'anchor '+p; s=s.replace(old,new,1); open(p,'w').write(s); print(p.split('/')[-1],'patched')
