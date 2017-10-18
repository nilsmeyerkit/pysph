#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=3G         # memory; default unit is megabytes
#SBATCH --time=0-20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
export OMP_NUM_THREADS=32

# changing to scratch directory
mkdir /scratch/nmeyer7/ar20_phi_$1
cd /scratch/nmeyer7/ar20_phi_$1
# running problem with openmp
pysph run fiber.rve --volfrac $1 --ar 20 --folgartucker --openmp
