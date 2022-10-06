dataFolder=../integration_spm/
modelFolder1=../Model_1

# Plot correlation
python plotCorrelationPINNvsData.py -v -mf $modelFolder1 -df $dataFolder -p 0.5 1
python plotCorrelationPINNvsData.py -v -mf $modelFolder1 -df $dataFolder -p 0.5 10
python plotCorrelationPINNvsData.py -v -mf $modelFolder1 -df $dataFolder -p 4 1
python plotCorrelationPINNvsData.py -v -mf $modelFolder1 -df $dataFolder -p 4 10
python plotCorrelationPINNvsData.py -v -mf $modelFolder1 -df $dataFolder -p 2 2





