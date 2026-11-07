import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'self.follow_mode' in s: print(p.split('/')[-1],'already'); continue
    s=s.replace("""PASS_MAX_RANGE          = 2.5      # [m]  only plan a pass when the opponent is closer than this""","""PASS_MAX_RANGE          = 2.5      # [m]  only plan a pass when the opponent is closer than this
FOLLOW_LATERAL_WINDOW   = 0.60     # [m]  a moving car within this lateral offset with no passing room -> follow straight behind it""",1)
    s=s.replace("""        self.pass_side = ''          # 'L' / 'R' while committed to a pass""","""        self.pass_side = ''          # 'L' / 'R' while committed to a pass
        self.follow_mode = False     # no room to pass: follow directly behind the car ahead""",1)
    # decision in apply_pass_mask
    old="""        now = time.monotonic()
        if not PASS_ENABLED:
            return
        if self.opponent_blocking() and self.opp_obs[0] < PASS_MAX_RANGE:"""
    new="""        now = time.monotonic()
        self.follow_mode = False
        if not PASS_ENABLED:
            return
        moving_ahead = (self.opp_obs is not None and self.opp_speed is not None and self.opp_count >= OPP_CONFIRM_SCANS
                        and self.opp_speed >= OPP_STATIONARY_SPEED and self.opp_obs[0] < PASS_MAX_RANGE
                        and abs(self.opp_obs[1]) <= FOLLOW_LATERAL_WINDOW)
        if moving_ahead:"""
    assert old in s; s=s.replace(old,new,1)
    old2="""                if max(self.free_left, self.free_right) >= PASS_MIN_WIDTH:
                    self.pass_side = side
                    self.pass_until = now + PASS_COMMIT_TIME
                else:
                    self.pass_side = ''"""
    new2="""                if max(self.free_left, self.free_right) >= PASS_MIN_WIDTH:
                    self.pass_side = side
                    self.pass_until = now + PASS_COMMIT_TIME
                else:
                    self.pass_side = ''
            if not self.pass_side:
                # no room on either side: do not squeeze into a slot next to it, follow straight behind
                self.follow_mode = True
                self.ranges[:] = 0.0
                return"""
    assert old2 in s; s=s.replace(old2,new2,1)
    # fallback + trailing use follow_mode too
    s=s.replace("""        if len(biggest_gap) == 0 and self.opponent_blocking():""","""        if len(biggest_gap) == 0 and (self.opponent_blocking() or self.follow_mode):""",1)
    old3="""        if not self.opponent_blocking() or self.pass_side:
            return speed  # nothing in the way, a stopped car (go around), or we are overtaking"""
    new3="""        if not (self.opponent_blocking() or self.follow_mode) or self.pass_side:
            return speed  # nothing in the way, a stopped car (go around), or we are overtaking"""
    assert old3 in s; s=s.replace(old3,new3,1)
    s=s.replace("""{" PASS-" + self.pass_side if self.pass_side else ""}""","""{" PASS-" + self.pass_side if self.pass_side else ""}{" FOLLOW" if self.follow_mode else ""}""",1)
    open(p,'w').write(s); print(p.split('/')[-1],'patched')
