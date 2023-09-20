dataFolder=integration_spm
rm -r Model*
rm -r Log*
python parameter_optimization.py -nosimp -df $dataFolder -i input_param_opt -opt --save-ckpt
