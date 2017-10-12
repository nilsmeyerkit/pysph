#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2.5G         # memory; default unit is megabytes
#SBATCH --time=0-5:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=16               # number of MPI processes

# changing to scratch directory
mkdir /scratch/nmeyer7/phi=$1
cd /scratch/nmeyer7/phi=$1
# running problem with openmp
pysph run fiber.rve --dim 3 --E 1E9 --volfrac $1 --ar 20 --D 10 --folgartucker --openmp
