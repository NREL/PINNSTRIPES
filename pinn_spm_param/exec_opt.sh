dataFolder=./integration_spm/

rm -r Model*
rm -r Log*


python main.py -opt -df $dataFolder -i input
