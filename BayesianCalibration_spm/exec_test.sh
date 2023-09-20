params="0.5 1"
utilFolder=../pinn_spm_param/util
dataFolder=../pinn_spm_param/integration_spm
modelFolder=../pinn_spm_param/tests/Model_1

python makeData.py -nosimp  -df $dataFolder -uf $utilFolder -n_t 100 -noise 0 -p $params
python makeData.py -nosimp  -df $dataFolder -uf $utilFolder -n_t 100 -noise 0.003 -p $params

# Plot correlation
python cal_nosigma.py -minsigma 0.001 -n_try 2 -uf $utilFolder -nosimp -mf $modelFolder -nt 100 -noise 0 -mcmc hmc
python cal_nosigma.py -minsigma 0.001 -n_try 2 -uf $utilFolder -nosimp -mf $modelFolder -nt 100 -noise 0.003 -mcmc hmc
