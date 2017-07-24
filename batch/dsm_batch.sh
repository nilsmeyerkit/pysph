#!/bin/bash

echo "########################################"
echo "#Running batch mode for DSM Parameters.#"
echo "########################################"

E=2.5E9
w=150
d=0.0001
G=3.3
mu=63
# ensuring that viscosity is still smallest step
massscale=0.5E5

echo "#### Aspect ratios ####"

echo "Running aspect ratio: 17 ..."
pysph ebg.channel --ar 17 --d $d --E $E --G $G --mu $mu --massscale $massscale --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/17particles/

echo "Running aspect ratio: 35 ..."
pysph ebg.channel --ar 35 --d $d --E $E --G $G --mu $mu --massscale $massscale --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/35particles/

echo "Running aspect ratio: 61 ..."
pysph ebg.channel --ar 61 --d $d --E $E --G $G --mu $mu --massscale $massscale --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_DSM/61particles/
