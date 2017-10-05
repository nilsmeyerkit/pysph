#!/bin/bash

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

for phi in 0.1 0.3 0.5
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/phi=${phi}
  cd /scratch/nmeyer7/phi=${phi}

  MEM=$((phi*25))

  # running problem with openmp
  sqsub -q threaded -n 16 -o /home/nmeyer7/phi=${phi}.log -r 10h --mpp ${MEM}G pysph run fiber.rve --dim 3 --volfrac ${phi} --ar 11 --massscale 1E8 --D 10 --folgartucker --openmp
done
