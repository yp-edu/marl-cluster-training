#!/bin/bash

JID_JOB1=`sbatch launch/exp:multiwalker-jz.sh 10000000 | cut -d " " -f 4`
sbatch  --dependency=afterok:$JID_JOB1 launch/exp:multiwalker-jz.sh 20000000 $JID_JOB1 10000000
