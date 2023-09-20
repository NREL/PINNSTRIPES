#!/bin/sh
dataFolder=./integration_spm/
input_dir=input_sweep
log_dir=sweep_log
mkdir -p $log_dir
nSim=`ls $input_dir | wc -l`

for (( isim=0; isim<nSim; isim++ ))
do
   echo Doing isim $isim
   python main.py -nosimp -df $dataFolder -opt  -i $input_dir/input$isim > $log_dir/log$isim
done

