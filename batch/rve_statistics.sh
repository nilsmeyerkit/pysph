#!/bin/bash

volfrac=0.01

for N in {2..10}
do
  pysph run fiber.rve --volfrac $volfrac --folgartucker --openmp
  mkdir sph-paper/results/volfrac1/${N}
  cp rve_output/N.csv sph-paper/results/volfrac1/${N}/N.csv
  cp rve_output/orientation.pdf sph-paper/results/volfrac1/${N}/orientation.pdf
  cp rve_output/viscosity.pdf sph-paper/results/volfrac1/${N}/viscosity.pdf
done
