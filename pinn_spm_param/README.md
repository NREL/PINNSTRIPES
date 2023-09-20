# PINN for Simpler single particle model (SPM)

## Installing

1. `conda create --name pinnstripes python=3.10`
2. `conda activate pinnstripes`
3. `pip install -r ../requirements.txt`

## Using the code

1. `preProcess`: make the data from finite difference integration. Do `bash exec.sh`

2. `./`: the main script is `main.py` that starts the training. Do `bash exec_opt.sh` for training in production mode. Do `bash exec_noOpt.sh` for training in debug mode. 

3. `postProcess`: post process the PINN training result. Link the correct model and log folder and do `bash exec.sh`

## Precision

`bash convert_to_float32.sh` will enable training in single precision mode.

`bash convert_to_float64.sh` will enable training in double precision mode.

## Capabilities

### Two stage training

The training occurs in two stage. First we use a SGD training and next a LBFGS training (for refinement). The number of epochs for SGD training and LBFGS training can be controlled via `EPOCHS` and `EPOCHS_LBFGS`. The SGD occurs by batches but the LBFGS does one step per epoch. To avoid memory issues, LBFGS can be set to accumulate the gradient by batches and then do use the accumulated gradient once all the batches are processed. The number of batches for SGD is controled with `N_BATCH` and for LBFGS with `N_BATCH_LBFGS`. 

Two stage training is activated if `LBFGS` is `True` in `main.py`

### PINN Losses

We use 4 different PINN losses

1. Interior losses that compute the residual of governing equations at interior collocations points. Number of collocation points by batch is controlled with `BATCH_SIZE_INT`.
2. Boundary losses that compute the residual of boundary conditions at boundary collocation points. Number of collocation points by batch is controlled with `BATCH_SIZE_BOUND`.
3. Data losses that evaluate the mismatch with provided data. The maximum batch size for data is controlled with `MAX_BATCH_SIZE_DATA`. If this number is too low, the code will ignore some of the data. The amount of data ignored is printed at the beginning of training.
4. Regularization losses that compute regularization conditions at reuglarization collocation points. Number of collocation points by batch is controlled with `BATCH_SIZE_REG`. Some regularization loss require performing an integration. The number of points used for integrating over each domain is controlled by `BATCH_SIZE_STRUCT`.
In the SPM case, no regularization is used.

The user may activate or deactivate each loss via the `alpha` parameter in `main.py`. The active or inactive losses are printed at the beginning of training.

### Learning rate

Learning rate is set with 2 parameters `LEARNING_RATE_MODEL` and `LEARNING_RATE_MODEL_FINAL`. The solver does half the epochs with the learning rate set with `LEARNING_RATE_MODEL` and then decays exponentially to `LEARNING_RATE_MODEL_FINAL`.

Similar workflow is set for the self-attention weights using `LEARNING_RATE_WEIGHTS` and `LEARNING_RATE_WEIGHTS_FINAL`.

The LBFGS learning rate can be set with `LEARNING_RATE_LBFGS`. 

### Battery model treatment

#### Hard enforcing of initial conditions

Initial conditions are enforced hardly. The rate at which we allow the neural net to deviate from the IC is given by `HARD_IC_TIMESCALE`. 

#### Exponential limiter

To avoid exploding gradients from the Butler Volmer relation, we clip interior of exponentials using `EXP_LIMITER`. Not used for SPM

#### J linearization

An option to linearize the `sinh` to a linear function can be activated with `LINEARIZE_J`. This typically allows not having very large losses at the beginning of training.

### Collocation mode

The user can either demand a `fixed` set of collocation points carried through the training, or demand that collocation points are `random`, i.e. randomly generated for each training step.

Random collocation is incompatible with self-attention weights.

### PINN training regularization

We can ask the PINN to gradually increase the time spanned by the collocation points via `GRADUAL_TIME`. If the user used `fixed` collocation mode, then the collocation points location are gradually stretched over time only.


We can use self-attention weights with the flag `DYNAMIC_ATTENTION_WEIGHTS`. Each collocation point and data points are then assigned a weight that is trained at each step of the SGD process.

### Neural net architecture

Activation can be piloted via `ACTIVATION`. Available options are 
- `tanh`
- `swish`
- `sigmoid`

Two architectures are available. Either a split architecture is used where each variable is determined by a separate branch. Otherwise a merge architecture is available where a common latent space to all variable that depend of `[t]`, `[t,x]`, `[t,r]` or `[t,x,r]` are constructed. Choose architecture with `MERGED` flag. Number of hidden layers can be decided with `hidden_unitsXXX` flags in `main.py`.


 

