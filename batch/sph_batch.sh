#!/bin/bash

echo "#####################"
echo "#Running batch mode.#"
echo "#####################"

# echo "#### Young's modulus ####"
#
# echo "# Running modulus: 1E7 ... #"
# pysph run ebg.channel --ar 21 --E 1E07 --holdcenter --openmp
# mv channel_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E07/
#
# echo "# Running modulus: 1E8 ... #"
# pysph run ebg.channel --ar 21 --E 1E08 --holdcenter --openmp
# mv channel_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E08/
#
# echo "# Running modulus: 1E9 ... #"
# pysph run ebg.channel --ar 21 --E 1E09 --holdcenter --openmp
# mv channel_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles_E=1E09/

echo "#### Aspect ratios ####"

echo "Running aspect ratio: 5 ..."
pysph run fiber.channel --ar 5 --holdcenter
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/5particles/

echo "Running aspect ratio: 11 ..."
pysph run fiber.channel --ar 11 --holdcenter
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/11particles/

echo "Running aspect ratio: 21 ..."
pysph run fiber.channel --ar 21 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/21particles/

echo "Running aspect ratio: 31 ..."
pysph run fiber.channel --ar 31 --holdcenter --openmp
mv channel_output/ \
~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/31particles/

# echo "Running aspect ratio: 41 ..."
# pysph run fiber.channel --ar 41 --holdcenter --openmp
# mv channel_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Shearflow_Gleb/41particles/
