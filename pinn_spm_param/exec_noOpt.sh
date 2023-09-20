dataFolder=./integration_spm/

rm -r Model*
rm -r Log*


python main.py -nosimp -df $dataFolder -i input
