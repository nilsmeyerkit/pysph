#!/bin/bash
#SBATCH --account=def-hrymak-ab
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4000M
#SBATCH --output=%x-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for phi in 0.05 0.1
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/phi=${phi}
  cd /scratch/nmeyer7/phi=${phi}

  # running problem with openmp
  sbatch pysph run fiber.rve --dim 3 --volfrac ${phi} --ar 11 --D 2 --folgartucker --openmp
done
