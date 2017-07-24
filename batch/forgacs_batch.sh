#!/bin/bash

echo "#####################"
echo "#Running batch mode.#"
echo "#####################"

pysph run ebg.channel --ar 171 --E 6.3E09 --d 0.0000122 --mu 9.12 --G 1 --openmp
mv shear_fiber_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Forgacs/171particles_G=1/
