#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=32               # number of MPI processes

##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#export OMP_NUM_THREADS=16

# changing to scratch directory
mkdir /scratch/nmeyer7/AR$1
cd /scratch/nmeyer7/AR$1

pysph run fiber.channel --ar $1 --E 1E11 --holdcenter --openmp
