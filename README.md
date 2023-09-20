# PINNSTRIPES (Physics-Informed Neural Network SurrogaTe for Rapidly Identifying Parameters in Energy Systems) [![PINNSTRIPES-CI](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml)

## Installing

1. `conda create --name pinnstripes python=3.10`
2. `conda activate pinnstripes`
3. `pip install -r requirements.txt`

## PINN for Li-ion Battery Single Particle Model (SPM)

Located in `pinn_spm_param`

### Quick start

1. `pinn_spm_param/preProcess`: make the data from finite difference integration. Do `bash exec.sh`

2. `pinn_spm_param`: the main script is `main.py` that starts the training. Do `bash exec_opt.sh` for training in production mode. Do `bash exec_noOpt.sh` for training in debug mode. 

3. `pinn_spm_param/postProcess`: post process the PINN training result. Link the correct model and log folder in `exec.sh` and do `bash exec.sh`

Also consider looking at the test suite in `pinn_spm_param/tests` and `.github/workflows/ci.yml` to understand how to use the code

### Precision

`cd pinn_spm_param`

`bash convert_to_float32.sh` will enable training in single precision mode.

`bash convert_to_float64.sh` will enable training in double precision mode.

### Simple or realistic parameter set

`cd pinn_spm_param`

`bash convert_to_simp.sh` will enable training with a simple electrochemical and transport properties

`bash convert_to_float64.sh` will enable training with realistic electrochemical and transport properties

`Data` contains experimental measurements of Uocp that are used to generate the Uocp functions in `pinn_spm_param/util/generateOCP.py` and `pinn_spm_param/util/generateOCP_poly_mon.py` 


### Two stage training

The training occurs in two stage. First we use a SGD training and next a LBFGS training (for refinement). The number of epochs for SGD training and LBFGS training can be controlled via `EPOCHS` and `EPOCHS_LBFGS`. The SGD occurs by batches but the LBFGS does one step per epoch. To avoid memory issues, LBFGS can be set to accumulate the gradient by batches and then do use the accumulated gradient once all the batches are processed. The number of batches for SGD is controled with `N_BATCH` and for LBFGS with `N_BATCH_LBFGS`. A warm start period can be used for `LBFGS` via `EPOCHS_START_LBFGS` which typically prevents blowup after the first steps.

`SGD` can be deactivated by setting `EPOCHS: 0` or by setting `SGD: False` in `pinn_spm_param/input`

`LBFGS` can be deactivated by setting `EPOCHS_LBFGS: 0` or by setting `LBFGS: False` in `pinn_spm_param/input`

### PINN Losses

We use 4 different PINN losses

1. Interior losses that compute the residual of governing equations at interior collocations points. Number of collocation points by batch is controlled with `BATCH_SIZE_INT`.
2. Boundary losses that compute the residual of boundary conditions at boundary collocation points. Number of collocation points by batch is controlled with `BATCH_SIZE_BOUND`.
3. Data losses that evaluate the mismatch with provided data. The maximum batch size for data is controlled with `MAX_BATCH_SIZE_DATA`. If this number is too low, the code will ignore some of the data. The amount of data ignored is printed at the beginning of training.
4. Regularization losses that compute regularization conditions at reuglarization collocation points. Number of collocation points by batch is controlled with `BATCH_SIZE_REG`. Some regularization loss require performing an integration. The number of points used for integrating over each domain is controlled by `BATCH_SIZE_STRUCT`.
In the SPM case, no regularization is used.

The user may activate or deactivate each loss via the `alpha` parameter in `main.py`. The active or inactive losses are printed at the beginning of training.

### PINN Losses weighting

The 4 PINN losses can be independently weighted via `alpha : 1.0 1.0 0.0 0.0`. In order, these coefficients weight the interior loss, the boundary loss, the data loss and the regularization loss.

Individual physics loss can be weighted via `w_phie_int` and others `w_xxx_xxx`

### Learning rate

Learning rate is set with 2 parameters `LEARNING_RATE_MODEL` and `LEARNING_RATE_MODEL_FINAL`. The solver does half the epochs with the learning rate set with `LEARNING_RATE_MODEL` and then decays exponentially to `LEARNING_RATE_MODEL_FINAL`.

Similar workflow is set for the self-attention weights using `LEARNING_RATE_WEIGHTS` and `LEARNING_RATE_WEIGHTS_FINAL`.

The LBFGS learning rate can be set with `LEARNING_RATE_LBFGS`. The learning rate for LBFGS is dynamically adjusted during training to avoid instabilities. The learning rate given in the input file is a target that LBFGS tries to attain.

### Battery model treatment

#### Hard enforcing of initial conditions

Initial conditions are strictly enforced. The rate at which we allow the neural net to deviate from the IC is given by `HARD_IC_TIMESCALE`. 

#### Exponential limiter

To avoid exploding gradients from the Butler Volmer relation, we clip interior of exponentials using `EXP_LIMITER`. Not used for SPM

#### J linearization

An option to linearize the `sinh` to a linear function can be activated with `LINEARIZE_J`. This typically allows not having very large losses at the beginning of training.

### Collocation mode

The user can either demand a `fixed` set of collocation points carried through the training, or demand that collocation points are `random`, i.e. randomly generated for each training step. Once LBFGS starts, the collocation points are held fixed.

Random collocation is incompatible with self-attention weights.

### PINN training regularization

We can ask the PINN to gradually increase the time spanned by the collocation points via `GRADUAL_TIME`. If the user used `fixed` collocation mode, then the collocation points location are gradually stretched over time only. The stretching schedule is controlled by `RATIO_FIRST_TIME : 1`. The stretching is done so that it reaches maximal time at the mid point in the SGD Epoch.

Likewise, the time interval can gradually increase during the LBFGS training. If the user sets a warmstart epoch number for LBFGS, the warm start is redone everytime the time interval increases. The time interval stretching is controlled via `N_GRADUAL_STEPS_LBFGS : 10` and `GRADUAL_TIME_MODE_LBFGS: exponential`. The time mode can be linear or exponential.

We can use self-attention weights with the flag `DYNAMIC_ATTENTION_WEIGHTS`. Each collocation point and data points are then assigned a weight that is trained at each step of the SGD process. The user can decide when to start weight training by adjusting `START_WEIGHT_TRAINING_EPOCH`. 

Weight annealing can be used by setting `ANNEALING_WEIGHTS: True`

### Neural net architecture

Activation can be piloted via `ACTIVATION`. Available options are 
- `tanh`
- `swish`
- `sigmoid`
- `elu`
- `selu`
- `gelu`


Two architectures are available. Either a split architecture is used where each variable is determined by a separate branch. Otherwise a merge architecture is available where a common latent space to all variable that depend of `[t]`, `[t,x]`, `[t,r]` or `[t,x,r]` are constructed. Choose architecture with `MERGED` flag. Number of hidden layers can be decided with `hidden_unitsXXX` flags in `main.py`.
Lastly gradient pathology blocks, residual blocks or fully connected blocks can be used.



### Hierarchy

The models can be used in hierchical modes by setting `HNN` and `HNNTIME`. Examples are available in `pinn_spm_param/tests`. The hierarchy can be done by training models over the same spatio temporal and parametric domain. It can be done by training lower hierarchy level up until a threshold time. It can be done by training the lower hierarchy levels for a specific parameter set.


## Bayesian calibration

The Bayesian calibration module only needs to call the trained model which can be passed via the command line. See `BayesianCalibration_spm/exec_test.sh`

The observational data can be generated via `makeData.py` (see `BayesianCalibration_spm/exec_test.sh` for usage)

The calibration can be done via `cal_nosigma.py` (see `BayesianCalibration_spm/exec_test.sh` for usage)

The likelihood uncertainty `sigma` is set via bisectional hyper parameter search.

## Formatting

Code formatting and import sorting is done automatically with `black` and `isort`.

Fix import and format : `bash fixFormat.sh`

Pushes and PR to `main` branch are forbidden without first running these commands

## References

SWR-22-12


Recommended citations

```

@article{j1,
  title={},
  author={},
  year={2023},
  institution={}
}

@article{j2,
  title={},
  author={},
  year={2023},
  institution={}
}

```




