dataFolder=../integration_spm/
modelFolder=../Model_1
logFolder=../Log_1

# Plot correlation
python plotData.py -nosimp -df $dataFolder -p 0.5 1

# Plot correlation
python plotCorrelationPINNvsData.py -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot contour of PINN prediction
python plotPINNResult.py -nosimp -mf $modelFolder -p 0.5 1

python plotPINNResult_movie.py -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot residuals
python plotResidualVariation.py -lf $logFolder -mf $modelFolder
