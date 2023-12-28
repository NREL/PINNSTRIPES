# PINNSTRIPES (Physics-Informed Neural Network SurrogaTe for Rapidly Identifying Parameters in Energy Systems) [![PINNSTRIPES-CI](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml) 

## Installing

1. `conda create --name pinnstripes python=3.10`
2. `conda activate pinnstripes`
3. `pip install -r requirements.txt`

## PINN for Li-ion Battery Single Particle Model (SPM)

Located in `pinn_spm_param`

### Quick start

1. `pinn_spm_param/preProcess`: make the data from finite difference integration. Do `bash exec.sh`

2. `pinn_spm_param`: the main script is `main.py` which starts the training. Do `bash exec_opt.sh` for training in production mode. Do `bash exec_noOpt.sh` for training in debug mode. 

3. `pinn_spm_param/postProcess`: post-process the PINN training result. Link the correct model and log folder in `exec.sh` and do `bash exec.sh`

Consider looking at the test suite in `pinn_spm_param/tests`, `BayesianCalibration_spm/exec_test.sh`, and `.github/workflows/ci.yml` to understand how to use the code

### Precision

`cd pinn_spm_param`

`bash convert_to_float32.sh` will enable training in single precision mode.

`bash convert_to_float64.sh` will enable training in double precision mode.

### Simple or realistic parameter set

`cd pinn_spm_param`

`bash convert_to_simp.sh` will enable training with simple electrochemical and transport properties

`bash convert_to_float64.sh` will enable training with realistic electrochemical and transport properties

`Data` contains experimental measurements of Uocp that are used to generate the Uocp functions in `pinn_spm_param/util/generateOCP.py` and `pinn_spm_param/util/generateOCP_poly_mon.py` 


### Two-stage training

The training occurs in two stages. First, we use SGD training, and next LBFGS training (for refinement). The number of epochs for SGD training and LBFGS training can be controlled via `EPOCHS` and `EPOCHS_LBFGS`. The SGD occurs in batches but the LBFGS does one step per epoch. To avoid memory issues, LBFGS can be set to accumulate the gradient by batches and then do use the accumulated gradient once all the batches are processed. The number of batches for SGD is controlled with `N_BATCH` and for LBFGS with `N_BATCH_LBFGS`. A warm start period can be used for `LBFGS` via `EPOCHS_START_LBFGS` which typically prevents blowup after the first steps.

`SGD` can be deactivated by setting `EPOCHS: 0` or by setting `SGD: False` in `pinn_spm_param/input`

`LBFGS` can be deactivated by setting `EPOCHS_LBFGS: 0` or by setting `LBFGS: False` in `pinn_spm_param/input`

### PINN losses

We use 4 different PINN losses

1. Interior losses compute the residual of governing equations at interior collocation points. The number of collocation points by batch is controlled with `BATCH_SIZE_INT`.
2. Boundary losses compute the residual of boundary conditions at boundary collocation points. The number of collocation points by batch is controlled with `BATCH_SIZE_BOUND`.
3. Data losses evaluate the mismatch with provided data. The maximum batch size for data is controlled with `MAX_BATCH_SIZE_DATA`. If this number is too low, the code will ignore some of the data. The amount of data ignored is printed at the beginning of training.
4. Regularization losses compute regularization conditions at regularization collocation points. The number of collocation points by batch is controlled with `BATCH_SIZE_REG`. Some regularization loss require performing an integration. The number of points used for integrating over each domain is controlled by `BATCH_SIZE_STRUCT`.
In the SPM case, no regularization is used.

The user may activate or deactivate each loss via the `alpha` parameter in `main.py`. The active or inactive losses are printed at the beginning of training.

### PINN losses weighting

The 4 PINN losses can be independently weighted via `alpha : 1.0 1.0 0.0 0.0`. In order, these coefficients weigh the interior loss, the boundary loss, the data loss, and the regularization loss.

Individual physics loss can be weighted via `w_phie_int` and others `w_xxx_xxx`

### Learning rate

Learning rate is set with 2 parameters `LEARNING_RATE_MODEL` and `LEARNING_RATE_MODEL_FINAL`. The solver does half the epochs with the learning rate set with `LEARNING_RATE_MODEL` and then decays exponentially to `LEARNING_RATE_MODEL_FINAL`.

A similar workflow is set for the self-attention weights using `LEARNING_RATE_WEIGHTS` and `LEARNING_RATE_WEIGHTS_FINAL`.

The LBFGS learning rate can be set with `LEARNING_RATE_LBFGS`. The learning rate for LBFGS is dynamically adjusted during training to avoid instabilities. The learning rate given in the input file is a target that LBFGS tries to attain.

### Battery model treatment

#### Strict enforcement of initial conditions

Initial conditions are strictly enforced. The rate at which we allow the neural net to deviate from the IC is given by `HARD_IC_TIMESCALE`. 

#### Exponential limiter

To avoid exploding gradients from the Butler Volmer relation, we clip the interior of exponentials using `EXP_LIMITER`. Not used for SPM

#### J linearization

An option to linearize the `sinh` to a linear function can be activated with `LINEARIZE_J`. This typically allows not having very large losses at the beginning of training.

### Collocation mode

The user can either demand a `fixed` set of collocation points carried through the training or demand that collocation points be `random`, i.e. randomly generated for each training step. Once LBFGS starts, the collocation points are held fixed.

Random collocation is incompatible with self-attention weights.

### PINN training regularization

We can ask the PINN to gradually increase the time spanned by the collocation points via `GRADUAL_TIME`. If `fixed` collocation mode is used, then the collocation points' locations are gradually stretched over time only. The stretching schedule is controlled by `RATIO_FIRST_TIME : 1`. The stretching is done so that it reaches maximal time at the midpoint in the SGD Epoch.

Likewise, the time interval can gradually increase during the LBFGS training. If the user sets a warm start epoch number for LBFGS, the warm start is redone every time the time interval increases. The time interval stretching is controlled via `N_GRADUAL_STEPS_LBFGS : 10` and `GRADUAL_TIME_MODE_LBFGS: exponential`. The time mode can be linear or exponential.

We can use self-attention weights with the flag `DYNAMIC_ATTENTION_WEIGHTS`. Each collocation point and data point is assigned a weight that is trained at each step of the SGD process. The user can decide when to start weight training by adjusting `START_WEIGHT_TRAINING_EPOCH`. 

Weight annealing can be used by setting `ANNEALING_WEIGHTS: True`

### Neural net architecture

Activation can be piloted via `ACTIVATION`. Available options are 
- `tanh`
- `swish`
- `sigmoid`
- `elu`
- `selu`
- `gelu`


Two architectures are available. Either a split architecture is used where each variable is determined by a separate branch. Otherwise, a merged architecture is available where a common latent space to all variables that depend of `[t]`, `[t,x]`, `[t,r]`, or `[t,x,r]` is constructed. Choose architecture with `MERGED` flag. The number of hidden layers can be decided with `hidden_unitsXXX` flags in `main.py`.
Lastly, gradient pathology blocks, residual blocks, or fully connected blocks can be used.


### Hierarchy

The models can be used in hierarchical modes by setting `HNN` and `HNNTIME`. Examples are available in `pinn_spm_param/tests`. The hierarchy can be done by training models over the same spatio-temporal and parametric domain. It can be done by training lower hierarchy levels up until a threshold time. It can be done by training the lower hierarchy levels for a specific parameter set.

## SPM Preprocess

Under `pinn_spm_param/integration_spm` an implicit and an explicit integrator are provided to generate solutions of the SPM equations. 

Under `pinn_spm_param/integration_spm`, run `python main.py -nosimp -opt -lean` to generate a rapid example of the SPM solution.

Under `pinn_spm_param/preProcess`, run `python makeDataset_spm.py -nosimp -df ../integration_spm -frt 1` to generate a dataset usable by the PINN.

The implicit integration is recommended for fine spatial discretization due to the diffusive CFL constraint. For rapid integration using a coarse grid, the explicit integration will be preferable. The explicit integrator automatically adjusts the timestep based on the CFL constraint.

## SPM post process

Under `pinn_spm_param/postProcess` see `exec.sh` for all the post-processing tools available.

- `plotData.py` will plot the data generated from the PDE integrator
- `plotCorrelationPINNvsData.py` will show 45 degree plots to check the accuracy of the PINN againsts the PDE integrator
- `plotPINNResult.py` will plot the fields predicted by the best PINN
- `plotPINNResult_movie.py` will plot the field predicted by the best PINN and the correlation plots as movies to check the evolution of the predictions over epochs.
- `plotResidualVariation.py` will display the different losses to ensure that they are properly balanced.


## Bayesian calibration

The Bayesian calibration module only needs to call the trained model which can be passed via the command line. See `BayesianCalibration_spm/exec_test.sh`

The observational data can be generated via `makeData.py` (see `BayesianCalibration_spm/exec_test.sh` for usage)

The calibration can be done via `cal_nosigma.py` (see `BayesianCalibration_spm/exec_test.sh` for usage)

The likelihood uncertainty `sigma` is set via bisectional hyperparameter search.

## Formatting [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Code formatting and import sorting are done automatically with `black` and `isort`. 

Fix imports and format : `pip install black isort; bash fixFormat.sh`

Spelling is checked but not automatically fixed using `codespell`

## Acknowledgements
This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE's Vehicle Technologies Office (VTO) and Advanced Scientific Computing Research (ASCR) program. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the repository do not necessarily represent the views of the DOE or the U.S. Government.

## References

Recommended citations

```

@article{hassanaly2023Physics1,
  title={Physics-informed neural network surrogates of Li-ion battery models for
parametr inference. \\Part I: Implementation and multifidelity hierarchies for the single-particle model},
  author={Malik Hassanaly, Peter J. Weddle, Ryan N. King, Subhayan De, Alireza Doostan, Corey R. Randall, Eric J. Dufek, Andrew M. Colclasure, 
Kandler Smith},
  year={2023},
}

@article{hassanaly2023Physics2,
  title={Physics-informed neural network surrogates of Li-ion battery models for parametr inference. \\Part II: Regularization and application of the pseudo-2D model},
  author={Malik Hassanaly, Peter J. Weddle, Ryan N. King, Subhayan De, Alireza Doostan, Corey R. Randall, Eric J. Dufek, Andrew M. Colclasure, 
Kandler Smith},
  year={2023},
}

@misc{osti_2204976,
title = {PINNSTRIPES (Physics-Informed Neural Network SurrogaTe for Rapidly Identifying Parameters in Energy Systems) [SWR-22-12]},
author = {Hassanaly, Malik and Smith, Kandler and King, Ryan and Weddle, Peter and USDOE Office of Energy Efficiency and Renewable Energy and USDOE Office of Science},
abstractNote = {Energy systems models typically take the form of complex partial differential equations which make multiple forward calculations prohibitively expensive. Fast and data-efficient construction of surrogate models is of utmost importance for applications that require parameter exploration such as design optimization and Bayesian calibration. In presence of a large number of parameters, surrogate models that capture correct dependencies may be difficult to construct with traditional techniques. The issue is addressed here with the formulation of the surrogate model constructed via Physics-Informed Neural Networks (PINN) which capture the dependence with respect to the parameters to estimate, while using a limited amount of data. Since forward evaluations of the surrogate model are cheap, parameter exploration is made inexpensive, even when considering a large number of parameters.},
url = {https://www.osti.gov//servlets/purl/2204976},
doi = {10.11578/dc.20231106.1},
url = {https://www.osti.gov/biblio/2204976}, year = {2023},
month = {10},
note =
}

```




