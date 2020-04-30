#!/bin/bash

rot=0.01

mkdir results

volfrac=0.001

for N in {1..2}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot $rot
  mkdir results/volfrac01
  mkdir results/volfrac01/${N}
  cp -a rve_output results/volfrac01/${N}
done

volfrac=0.01

for N in {1..2}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot $rot
  mkdir results/volfrac1
  mkdir results/volfrac1/${N}
  cp -a rve_output results/volfrac1/${N}
done

volfrac=0.1

for N in {1..2}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot $rot
  mkdir results/volfrac10
  mkdir results/volfrac10/${N}
  cp -a rve_output results/volfrac10/${N}
done

volfrac=0.3

for N in {1..2}
do
  pysph run fiber.rve --volfrac $volfrac --openmp --rot $rot
  mkdir results/volfrac30
  mkdir results/volfrac30/${N}
  cp -a rve_output results/volfrac30/${N}
done
