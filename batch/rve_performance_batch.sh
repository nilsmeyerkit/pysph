#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=3G         # memory; default unit is megabytes
#SBATCH --time=0-5:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --nodes=1
#SBATCH --cpus-per-node=16
export OMP_NUM_THREADS=16

# changing to scratch directory
mkdir /scratch/nmeyer7/phi=$1
cd /scratch/nmeyer7/phi=$1
# running problem with openmp
pysph run fiber.rve --dim 3 --E 1E9 --volfrac $1 --ar 10 --D 2 --folgartucker --openmp
