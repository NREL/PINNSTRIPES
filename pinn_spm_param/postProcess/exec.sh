dataFolder=../integration_spm/
modelFolder=../ModelFin_1
logFolder=../LogFin_1

# Plot correlation
python plotData.py -v -mf $modelFolder -df $dataFolder

# Plot correlation
python plotCorrelationPINNvsData.py -v -mf $modelFolder -df $dataFolder

# Plot contour of PINN prediction
python plotPINNResult.py -v -mf $modelFolder -p 2 2

# Plot residuals
python plotResidualVariation.py -v -lf $logFolder -mf $modelFolder
