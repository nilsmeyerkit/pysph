#!/bin/bash

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

for phi in 0.1 0.3 0.5
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/phi=${phi}
  cd /scratch/nmeyer7/phi=${phi}

  # running problem with openmp
  sqsub -q threaded -n 16 -o /home/nmeyer7/phi=${phi}.log -r 10h --mpp 2.5G pysph run fiber.channel --dim 3 --volfrac ${phi} --ar 11 --massscale 1E8 --holdcenter --D 100 --openmp
done
