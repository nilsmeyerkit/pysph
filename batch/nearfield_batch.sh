#!/bin/bash

echo "#########################################"
echo "#Running batch mode for nearfield model.#"
echo "#########################################"

fluidres=1
pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/

pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/

fluidres=0.5
pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb --openmp
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/

pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --openmp
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/

fluidres=0.25
pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --pb --openmp
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}_pb/

pysph run ebg.nearfield --ar 20 --V 1E-4 --massscale 1E8 --fluidres $fluidres --openmp
mv nearfield_output/ \
~/Dropbox/Thesis/Documentation/SPH/Nearfield/Res_${fluidres}/
