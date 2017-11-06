#!/bin/bash
E=2.5E9
d=0.0001
G=3.3
mu=63

pysph run fiber.channel --rot 2 --D 10000 --ar 17 --d $d --E $E --G $G --mu $mu --holdcenter --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/17particles/

pysph run fiber.channel --rot 1 --D 10000 --ar 45 --d $d --E $E --G $G --mu $mu --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/45particles/

pysph run fiber.channel --rot 1 --D 10000 --ar 81 --d $d --E $E --G $G --mu $mu --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/81particles/
