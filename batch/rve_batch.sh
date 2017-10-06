#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --time=0-02:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=32               # number of MPI processes

# changing to scratch directory
mkdir /scratch/nmeyer7/phi=$1
cd /scratch/nmeyer7/phi=$1
# running problem with openmp
pysph run fiber.rve --dim 3 --volfrac $1 --ar 20 --D 5 --folgartucker --openmp
