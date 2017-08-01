#!/bin/bash

echo "#########################################"
echo "#Running batch mode for nearfield model.#"
echo "#########################################"

# setting up virtual python
cd /home/nmeyer7/virtual_python
source fake_venv.sh

# fluidres=1
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/
#
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/
#
# fluidres=0.5
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb --openmp
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/
#
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --openmp
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/
#
# fluidres=0.25
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb --openmp
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/
#
# pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --openmp
# mv nearfield_output/ \
# ~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/

fluidres=1
mkdir /scratch/nmeyer7/res${fluidres}
cd /scratch/nmeyer7/res${fluidres}
sqsub -o ~/nmeyer7/res${fluidres} -r 3h pysph run ebg.nearfield --ar 20 --g 10 --massscale 1E8 --fluidres $fluidres --pb
mv /scratch/nmeyer7/res${fluidres} ~/res${fluidres}

fluidres=0.5
mkdir /scratch/nmeyer7/res${fluidres}
cd /scratch/nmeyer7/res${fluidres}
sqsub -q threaded -n 4 -o ~/nmeyer7/res${fluidres} -r 3h pysph run ebg.nearfield --ar 20 --g 10 --massscale 1E8 --fluidres $fluidres --pb --openmp
mv /scratch/nmeyer7/res${fluidres} ~/res${fluidres}

fluidres=0.25
mkdir /scratch/nmeyer7/res${fluidres}
cd /scratch/nmeyer7/res${fluidres}
sqsub -q threaded -n 16 -o ~/nmeyer7/res${fluidres} -r 3h pysph run ebg.nearfield --ar 20 --g 10 --massscale 1E8 --fluidres $fluidres --pb --openmp
mv /scratch/nmeyer7/res${fluidres} ~/res${fluidres}
