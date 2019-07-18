#!/bin/bash

volfrac=0.001

for N in {6..6}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot 4
  mkdir sph-paper/results/volfrac01/${N}
  cp -ar rve_output sph-paper/results/volfrac01/${N}
done

volfrac=0.01

for N in {6..6}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot 4
  mkdir sph-paper/results/volfrac1/${N}
  cp -ar rve_output sph-paper/results/volfrac1/${N}
done

volfrac=0.1

for N in {6..6}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot 4
  mkdir sph-paper/results/volfrac10/${N}
  cp -ar rve_output sph-paper/results/volfrac10/${N}
done

volfrac=0.3

for N in {6..6}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot 4
  mkdir sph-paper/results/volfrac30/${N}
  cp -ar rve_output sph-paper/results/volfrac30/${N}
done
