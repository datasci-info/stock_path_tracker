#!/bin/bash

DIRS=('is_turnpt')
Ns=(4)
# DEPTHs=(5 10 20)
# L2_LEAF_REGs=(100 200 300 400)
# Learning_rates=(0.5 0.1 0.05)

DEPTHs=(15)
L2_LEAF_REGs=(400)
Learning_rates=(0.05)

for direction in "${DIRS[@]}"
do
    for depth in "${DEPTHs[@]}"
    do
        for l2_reg in "${L2_LEAF_REGs[@]}"
        do
            for learn_rate in "${Learning_rates[@]}"
            do
                for n in ${Ns[@]}
                do
                    export N=$n
                    export DIRECTION=$direction
                    export LEARNING_RATE=$learn_rate
                    export L2_LEAF_REG=$l2_reg
                    export DEPTH=$depth
                    python model.py  > models/${N}_${direction}_${depth}_${learn_rate}_${l2_reg}.log 2>&1 &
                done
            done
        done
    done
done

wait

for direction in "${DIRS[@]}"
do
    for n in ${Ns[@]}
    do
        tail -1 models/${N}_${direction}_${depth}_${learn_rate}_${l2_reg}.log >> model_result.csv
    done

done


