#!/bin/bash

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

for ar in 11 21 31 41 51 61
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/ar=${ar}
  cd /scratch/nmeyer7/ar=${ar}

  D=$(($ar*1000))
  M=$((1000000/$ar))

  # running problem with openmp
  sqsub -q threaded -n 16 -o /home/nmeyer7/ar=${ar}.log -r ${ar}h --mpp 2.5G pysph run fiber.channel --ar ${ar} --massscale ${M}--rot 10 --holdcenter --D ${D} --openmp
done
