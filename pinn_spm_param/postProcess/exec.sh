dataFolder=../integration_spm/
modelFolder=../ModelFin_1
logFolder=../LogFin_1

# Plot PDE solution
python plotData.py -nosimp -df $dataFolder -p 0.5 1

# Plot correlation PDE-PINN
python plotCorrelationPINNvsData.py -v -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot contour of PINN prediction
python plotPINNResult.py -v -nosimp -mf $modelFolder -p 0.5 1

# Plot contour of PINN prediction and correlation with PDE as a movie
python plotPINNResult_movie.py -nosimp -mf $modelFolder -df $dataFolder -p 0.5 1

# Plot residuals over training steps
python plotResidualVariation.py -lf $logFolder -mf $modelFolder
