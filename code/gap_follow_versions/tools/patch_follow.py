import sys
for p in sys.argv[1:]:
    s=open(p).read()
    if 'follow the car ahead instead of stopping' in s: print(p.split('/')[-1],'already'); continue
    old="""            self.ranges = unmasked
            biggest_gap = self.find_biggest_gap()"""
    new="""            self.ranges = unmasked
            biggest_gap = self.find_biggest_gap()
        if len(biggest_gap) == 0 and self.opponent_blocking():
            # no gap because a moving car fills the lane: follow the car ahead instead of stopping
            # (trailing then holds the gap); a stopped car / obstacle never takes this path
            opp_idx = int(np.argmin(np.abs(self.angles - math.atan2(self.opp_obs[1], self.opp_obs[0]))))
            self.ranges[opp_idx] = self.opp_dist
            biggest_gap = np.array([opp_idx])"""
    assert old in s; s=s.replace(old,new,1); open(p,'w').write(s); print(p.split('/')[-1],'patched')
