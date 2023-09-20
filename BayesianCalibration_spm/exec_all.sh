params="2 2"
utilFolder=pinn_spm_param_fullpar_hierarchy_fix2_data/util
dataFolder=pinn_spm_param_fullpar_hierarchy_fix2_data/integration_spm
modelFolder=pinn_spm_param_fullpar_hierarchy_fix2_data_finetune/ModelFin_0

python makeData.py -nosimp  -df $dataFolder -uf $utilFolder -n_t 100 -noise 0 -p $params
python makeData.py -nosimp  -df $dataFolder -uf $utilFolder -n_t 100 -noise 0.003 -p $params

# Plot correlation
python preprocess.py -nosimp -uf $utilFolder -mf $modelFolder -n_t 100 -df $dataFolder
python preprocess.py -nosimp -uf $utilFolder -mf $modelFolder -n_t 100 -noise 0.003 -df $dataFolder

# Plot correlation
python cal_nosigma.py -minsigma 0.001 -n_try 10 -uf $utilFolder -nosimp -mf $modelFolder -nt 100 -noise 0 -mcmc hmc
python cal_nosigma.py -minsigma 0.001 -n_try 10 -uf $utilFolder -nosimp -mf $modelFolder -nt 100 -noise 0.003 -mcmc hmc
