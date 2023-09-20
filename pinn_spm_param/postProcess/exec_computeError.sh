dataFolder=../integration_spm/
modelFolder=../ModelFin_1
logFolder=../LogFin_1

python computeManyMetrics.py -nosimp -df $dataFolder
python computeError.py -nosimp -mf $modelFolder -df $dataFolder

