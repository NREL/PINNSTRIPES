dataFolder=./integration_spm/

rm -r Model*
rm -r Log*


python main.py -nosimp -opt -df $dataFolder -i input
