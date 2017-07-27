#!/bin/bash

echo "#####################"
echo "#Running batch mode.#"
echo "#####################"

echo "### Running Oscillations ###"
E=1E8
N=20

for D in 0 0.5 1 2
do
  gx=10
  gy=0
  pysph run ebg.beam --gx $gx --gy $gy --D $D --N $N --E $E
  dirx=~/Dropbox/Thesis/Documentation/EBG/Batch/X-Oscillation/N=${N}gx=${gx}_gy=${gy}_D=${D}_E=${E}/
  mv beam_output/ $dirx
  plot_list_x+="${dirx}oscillation_${D}.csv "
  gx=0
  gy=10
  pysph run ebg.beam --gx $gx --gy $gy --D $D --N $N --E $E
  diry=~/Dropbox/Thesis/Documentation/EBG/Batch/Y-Oscillation/N=${N}gx=${gx}_gy=${gy}_D=${D}_E=${E}/
  mv beam_output/ $diry
  plot_list_y+="${diry}oscillation_${D}.csv "
done
python ~/Dropbox/Thesis/Documentation/EBG/Batch/plot_osci.py 'oscillation_x.png' ${plot_list_x[*]}
python ~/Dropbox/Thesis/Documentation/EBG/Batch/plot_osci.py 'oscillation_y.png' ${plot_list_y[*]}


echo "### Running Displacments ###"
plot_list_x=()
plot_list_y=()
E=1E8
D=1

for N in 10 20 100
do
  gx=10
  gy=0
  pysph run ebg.beam --gx $gx --gy $gy --D $D --N $N --E $E
  dirx=~/Dropbox/Thesis/Documentation/EBG/Batch/X-Displacement/N=${N}gx=${gx}_gy=${gy}_D=${D}_E=${E}/
  mv beam_output/ $dirx
  plot_list_x+="${dirx}disp_${N}.csv "
  gx=0
  gy=10
  pysph run ebg.beam --gx $gx --gy $gy --D $D --N $N --E $E
  diry=~/Dropbox/Thesis/Documentation/EBG/Batch/Y-Displacement/N=${N}gx=${gx}_gy=${gy}_D=${D}_E=${E}/
  mv beam_output/ $diry
  plot_list_y+="${diry}disp_${N}.csv "
done
python ~/Dropbox/Thesis/Documentation/EBG/Batch/plot_disp.py 'displacement_x.png' ${plot_list_x[*]}
python ~/Dropbox/Thesis/Documentation/EBG/Batch/plot_disp.py 'displacement_y.png' ${plot_list_y[*]}

echo "### Running Large Deformation Run ###"
plot_list_y=()
E=1E4
D=1

for N in 10 20 100
do
  gx=0
  gy=10
  pysph run ebg.beam --gx $gx --gy $gy --D $D --N $N --E $E
  diry=~/Dropbox/Thesis/Documentation/EBG/Batch/Y-Displacement/N=${N}gx=${gx}_gy=${gy}_D=${D}_E=${E}/
  mv beam_output/ $diry
  plot_list_y+="${diry}disp_${N}.csv "
done
python ../Documentation/EBG/Batch/plot_large_disp.py 'large_displacement_y.png' ${plot_list_y[*]}
