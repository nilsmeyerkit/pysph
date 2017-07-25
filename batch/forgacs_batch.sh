#!/bin/bash

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

# changing to scratch directory
cd /scratch/nmeyer7

# running problem with openmp
pysph run ebg.channel --ar 171 --E 6.3E09 --d 0.0000122 --mu 9.12 --G 1 --openmp
