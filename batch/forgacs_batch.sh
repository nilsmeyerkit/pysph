#!/bin/bash

# G = 1.0, 3.20, 3.54, 4.35
G=1.0

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

# changing to scratch directory
mkdir /scratch/nmeyer7/G=${G}
cd /scratch/nmeyer7/G=${G}

# running problem with openmp
sqsub -q threaded -n 4 -o ~/nmeyer7/G=${G} -r 8h --mpp 2G pysph run ebg.channel --ar 171 --E 6.3E09 --d 0.0000122 --mu 9.12 --G ${G} --holdcenter --openmp
