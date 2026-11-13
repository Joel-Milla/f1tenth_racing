#!/bin/bash
# usage: level.sh NAME LAT BRAKE MIN
F=/home/f1tenth/f1tenth_ws/src/gap_follow/gap_follow/gap_follow_params.py
S=/tmp/claude-1000/-home-f1tenth-f1tenth-ws/fee9f4c2-6d52-40c9-8f40-96439f544a19/scratchpad
cp $S/gap_follow_params.step1.py $F
sed -i -E -e "s/^LATERAL_ACCEL( +)= [0-9.]+ /LATERAL_ACCEL\1= $2 /" -e "s/^BRAKE_DECEL( +)= [0-9.]+ /BRAKE_DECEL\1= $3 /" -e "s/^MIN_SPEED( +)= [0-9.]+ /MIN_SPEED\1= $4 /" $F
[ -z "$5" ] || sed -i -E "s/^MAX_ACCEL( +)= [0-9.]+ /MAX_ACCEL\1= $5 /" $F
grep -E "^(MAX_ACCEL|LATERAL_ACCEL|BRAKE_DECEL|MIN_SPEED|MAX_SPEED) " $F | cut -c1-34
cp $F $S/gap_follow_params.$1.py
source /opt/ros/humble/setup.bash; cd $S
timeout -s INT 45 python $F > node_$1.log 2>&1 &
sleep 2; python3 rec2.py 40 >/dev/null; wait; cp rec2.pkl $1.pkl; python3 analyze.py $1.pkl
rm -rf /home/f1tenth/f1tenth_ws/src/gap_follow/gap_follow/__pycache__
