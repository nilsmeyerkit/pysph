#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --time=6-00:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=32               # number of MPI processes

# changing to scratch directory
mkdir /scratch/nmeyer7/G=$1
cd /scratch/nmeyer7/G=$1

# running problem with openmp
 pysph run fiber.channel --ar 171 --E 6.3E09 --d 0.0000122 --mu 9.12 --G $1 --holdcenter
