#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --ntasks=8               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-02:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log

for phi in 0.05 0.1
do
  # changing to scratch directory
  mkdir /scratch/nmeyer7/phi=${phi}
  cd /scratch/nmeyer7/phi=${phi}

  # running problem with openmp
  srun pysph run fiber.rve --dim 3 --volfrac ${phi} --ar 11 --D 2 --folgartucker --openmp
done
