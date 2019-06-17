#!/bin/bash

volfrac=0.003

for N in {1..10}
do
  pysph run fiber.rve --volfrac $volfrac --openmp
  mkdir sph-paper/results/volfrac1/${N}
  cp -ar rve_output sph-paper/results/volfrac1/${N}
done
