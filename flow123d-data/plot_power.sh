#!/bin/bash

dir=${1:-test_th}

#make plot of production power
echo "
set terminal pdf color enhanced lw 2
set output '${dir}_power.pdf'
set xlabel 'time [years]'
set ylabel 'power [MW]'
set y2label 'temperature [C]'
set ytics nomirror
set y2tics
plot [1:] \"< sed -E '/\\\\.(left_fr_left_well|left_well|right_fr_right_well|right_well)\\\"/p' $dir/energy_balance.txt | awk 'BEGIN {t=0;v=0} {if (t==\$1) v+=\$4; else {print t, v; t=\$1;v=\$4}} END {print t,v}'\" u (\$1/365.25/86400):(-\$2/1e6) w l t 'power',\\
          \"< python3 parse_temp.py $dir\" u (\$1/86400/365.25):(\$2-273.15) w l axes x1y2 t 'temperature'" | gnuplot

