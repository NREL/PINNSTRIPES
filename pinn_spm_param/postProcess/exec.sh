dataFolder=../integration_spm/
modelFolder=../ModelFin_1
logFolder=../LogFin_1

# Plot correlation
python plotData.py -nosimp -df $dataFolder -p 0.5 1

# Plot correlation
python plotCorrelationPINNvsData.py -v -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot contour of PINN prediction
python plotPINNResult.py -v -nosimp -mf $modelFolder -p 0.5 1

python plotPINNResult_movie.py -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot residuals
python plotResidualVariation.py -lf $logFolder -mf $modelFolder
