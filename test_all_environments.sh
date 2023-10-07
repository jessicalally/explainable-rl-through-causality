#!/bin/bash

OLDIFS="IFS"
IFS=','
for i in "cartpole","ddqn" "lunarlander","ddqn" "mountaincar","dqn" "taxi","sarsa"
do
    set -- $i
    echo $1 and $2
    python main.py --env $1 --rl $2
done
