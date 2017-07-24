#!/bin/bash

echo "#####################"
echo "#Running batch mode.#"
echo "#####################"

echo "#### Young's modulus ####"

echo "# Running modulus: 1E7 ... #"
pysph run ebg.channel --ar 21 --E 1E07 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E07/

echo "# Running modulus: 1E8 ... #"
pysph run ebg.channel --ar 21 --E 1E08 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E08/

echo "# Running modulus: 1E9 ... #"
pysph run ebg.channel --ar 21 --E 1E09 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E09/

echo "#### Aspect ratios ####"

echo "Running aspect ratio: 11 ..."
pysph run ebg.channel --ar 11 --holdcenter
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/11particles_E=1E11/

echo "Running aspect ratio: 21 ..."
pysph run ebg.channel --ar 21 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E11/

echo "Running aspect ratio: 41 ..."
pysph run ebg.channel --ar 41 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/41particles_E=1E11/

echo "Running aspect ratio: 61 ..."
pysph run ebg.channel --ar 61 --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/61particles_E=1E11/
