#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --time=0-03:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=16

# changing to scratch directory
mkdir /scratch/nmeyer7/nearfield_res_$1
cd /scratch/nmeyer7/nearfield_res_$1
pysph run fiber.channel --ar 1 --width 20 --holdcenter --g 10 --G 0 --massscale 1E8 --fluidres $1 --openmp
