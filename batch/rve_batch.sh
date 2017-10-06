#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --ntasks=8               # number of MPI processes
#SBATCH --mem-per-cpu=4G         # memory; default unit is megabytes
#SBATCH --time=0-02:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log

# changing to scratch directory
mkdir /scratch/nmeyer7/phi=$1
cd /scratch/nmeyer7/phi=$1
# running problem with openmp
pysph run fiber.rve --dim 3 --volfrac $1 --ar 11 --D 2 --folgartucker --openmp
