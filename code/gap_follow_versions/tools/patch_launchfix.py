import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'TRAIL_MIN_OWN_SPEED' in s: print(p.split('/')[-1],'already'); continue
    s=s.replace("""TRAIL_GAP               = 0.8""","""TRAIL_MIN_OWN_SPEED     = 0.3      # [m/s] no trailing while we are (nearly) stopped: lidar noise at standstill looked like
                                   #       a moving car and permanently cancelled the launch before the start
TRAIL_GAP               = 0.8""",1)
    old="""        return in_path and self.opp_speed >= OPP_STATIONARY_SPEED"""
    new="""        return in_path and self.opp_speed >= OPP_STATIONARY_SPEED and self.speed > TRAIL_MIN_OWN_SPEED"""
    assert old in s,'anchor1 '+p; s=s.replace(old,new,1)
    old2="""            self.trailing = True
            self.launch_done = True        # never launch into a car
            self.last_speed_cmd = trail_speed"""
    new2="""            self.trailing = True           # (launch is paused while trailing, not cancelled for good)
            self.last_speed_cmd = trail_speed"""
    assert old2 in s,'anchor2 '+p; s=s.replace(old2,new2,1)
    old3="""            if self.launch_active():
                self.publish_velocity(LAUNCH_SPEED, steering_angle)"""
    new3="""            if not self.trailing and self.launch_active():
                self.publish_velocity(LAUNCH_SPEED, steering_angle)"""
    assert old3 in s,'anchor3 '+p; s=s.replace(old3,new3,1)
    open(p,'w').write(s); print(p.split('/')[-1],'patched')
