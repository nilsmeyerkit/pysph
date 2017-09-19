#!/bin/bash

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

for ar in 31 41 51
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/ar=${ar}
  cd /scratch/nmeyer7/ar=${ar}

  # running problem with openmp
  sqsub -q threaded -n 16 -o /home/nmeyer7/ar=${ar}.log -r 10h --mpp 2.5G pysph run fiber.channel --ar ${ar} --rot 10 --holdcenter --openmp
done