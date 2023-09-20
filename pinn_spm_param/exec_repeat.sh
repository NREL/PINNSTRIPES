dataFolder=./integration_spm/
rm -r Model*
rm -r Log*
python repeat_sim.py -nosimp -opt -df $dataFolder -i input
