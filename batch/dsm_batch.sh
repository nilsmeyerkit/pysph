#!/bin/bash
#SBATCH --account=def-hrymak-ab  # account name
#SBATCH --mem-per-cpu=2G         # memory; default unit is megabytes
#SBATCH --time=0-02:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out       # output log (<filename>-<jobid>.out)
#SBATCH --ntasks=16               # number of MPI processes
E=2.5E9
w=150
d=0.0001
G=3.3
mu=63
massscale=0.5E5

# pysph run fiber.channel --rot 2 --ar 17 --d $d --E $E --G $G --mu $mu --holdcenter --openmp --vtk

pysph run fiber.channel --rot 1.5 --ar 41 --d $d --E $E --G $G --mu $mu --holdcenter --openmp --vtk

pysph run fiber.channel --rot 1.5 --ar 61 --d $d --E $E --G $G --mu $mu --holdcenter --openmp --vtk
