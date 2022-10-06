dataFolder=./integration_spm/

rm -r Model*
rm -r Log*


python main.py -df $dataFolder -i input
