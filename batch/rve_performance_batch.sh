#!/bin/bash
#SBATCH --account=def-hrymak-ab # account name
#SBATCH --mem-per-cpu=3G        # memory; default unit is megabytes
#SBATCH --time=0-16:00          # time (DD-HH:MM)
#SBATCH --output=%x-%j.out      # output log (<filename>-<jobid>.out)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# changing to scratch directory
mkdir /scratch/nmeyer7/${SLURM_CPUS_PER_TASK}cpu_phi$1
cd /scratch/nmeyer7/${SLURM_CPUS_PER_TASK}cpu_phi$1
# running problem with openmp
pysph run fiber.rve --volfrac $1 --rot 0.01 --ar 50 --disable-output --openmp
