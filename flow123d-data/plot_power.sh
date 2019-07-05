#!/bin/bash

#make plot of production power (sum of energy flux through well and fracture boundaries)
echo "
set terminal pdf color enhanced lw 2
set output 'power.pdf'
set xlabel 'time [years]'
plot [1:] \"< sed -E '/\\\\.(left_fr_left_well|left_well|right_fr_right_well|right_well)\\\"/p' output_th/energy_balance.txt | awk 'BEGIN {t=0;v=0} {if (t==\$1) v+=\$4; else {print t, v; t=\$1;v=\$4}} END {print t,v}'\" u (\$1/365.25/86400):(-\$2/1e6) w lp t 'power [MW]'" | gnuplot