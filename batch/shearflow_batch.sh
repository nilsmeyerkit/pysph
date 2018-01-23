#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=32               # number of MPI processes

# changing to scratch directory
mkdir /scratch/nmeyer7/shearflow/AR$1
cd /scratch/nmeyer7/shearflow/AR$1

if [$1 < 35]
then
  pysph run fiber.shearflow --ar $1 --holdcenter --openmp
else
  pysph run fiber.shearflow --ar $1 --rot 2 --openmp
fi
