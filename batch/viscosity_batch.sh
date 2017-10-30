#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=3G         # memory; default unit is megabytes
#SBATCH --time=0-04:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
export OMP_NUM_THREADS=8

# changing to scratch directory
mkdir /scratch/nmeyer7/ar10_phi_$1
cd /scratch/nmeyer7/ar10_phi_$1
# running problem with openmp
pysph run fiber.rve --volfrac $1 --ar 10 --folgartucker --rot 10 --openmp
