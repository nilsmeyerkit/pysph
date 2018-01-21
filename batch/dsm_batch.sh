#!/bin/bash
E=2.5E9
d=0.0001
G=3.3
mu=63

pysph run fiber.channel --rot 2 --ar 11 --holdcenter --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/11particles/

pysph run fiber.channel --rot 2 --ar 21 --holdcenter --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/21particles/

pysph run fiber.channel --rot 1 --ar 45 --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/45particles/

pysph run fiber.channel --rot 1 --ar 81 --openmp --vtk
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/81particles/
