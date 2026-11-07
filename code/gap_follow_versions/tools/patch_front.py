import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'FRONT_MIN_X' in s: print(p.split('/')[-1],'already'); continue
    s=s.replace("""LIDAR_X_OFFSET          = 0.27""","""FRONT_MIN_X             = 0.15  # [m]  points must be at least this far in front of the lidar to count as "ahead" (front
                                #      bumper); walls BESIDE the car no longer read as an obstacle at 0 m (bad starts)
LIDAR_X_OFFSET          = 0.27""",1)
    n=0
    for old,new in [
        ("""        in_path = (arc_length > LIDAR_X_OFFSET) & (lateral_error <= SPEED_PATH_HALF_WIDTH)""",
         """        in_path = (arc_length > LIDAR_X_OFFSET + FRONT_MIN_X) & (lateral_error <= SPEED_PATH_HALF_WIDTH)"""),
        ("""        straight = (x > LIDAR_X_OFFSET) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)""",
         """        straight = (x > LIDAR_X_OFFSET + FRONT_MIN_X) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)"""),
        ("""            in_path = (fx > 0) & (np.abs(fy) <= SPEED_PATH_HALF_WIDTH)""",
         """            in_path = (fx > FRONT_MIN_X) & (np.abs(fy) <= SPEED_PATH_HALF_WIDTH)"""),
        ("""            ahead = (x > 0) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
            d_ahead = float(np.min(x[ahead])) if np.any(ahead) else SPEED_RANGE_CAP
            if d_ahead < LAUNCH_ABORT_DIST:""",
         """            ahead = (x > FRONT_MIN_X) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
            d_ahead = float(np.min(x[ahead])) if np.any(ahead) else SPEED_RANGE_CAP
            if d_ahead < LAUNCH_ABORT_DIST:"""),
        ("""            in_corridor = (sx > 0) & (np.abs(sy) <= STRAIGHT_LOCK_HALF_WIDTH)""",
         """            in_corridor = (sx > FRONT_MIN_X) & (np.abs(sy) <= STRAIGHT_LOCK_HALF_WIDTH)"""),
    ]:
        if old in s: s=s.replace(old,new,1); n+=1
    open(p,'w').write(s); print(p.split('/')[-1],'patched',n,'spots')
