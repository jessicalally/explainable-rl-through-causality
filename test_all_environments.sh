#!/bin/bash

OLDIFS="IFS"
IFS=','
for i in "starcraft","a2c" "cartpole","ddqn" "lunarlander","ddqn" "taxi","sarsa" "mountaincar","dqn"
do
    set -- $i
    echo $1 and $2
    python main.py --env $1 --rl $2
done