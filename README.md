# PINNSTRIPES (Physics-Informed Neural Network SurrogaTe for Rapidly Identifying Parameters in Energy Systems) [![PINN-CI](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/PINNSTRIPES/actions/workflows/ci.yml)

## Main codes

`pinn_spm_param`: PINN of the SPM model in discharge mode. Anode reaction rate (`i0_a`) and cathode diffusivity (`Ds_c`) are input parameters

## Calibration tools

`BayesianCalibration_spm` : Do Bayesian calibration with the surrogate and the PDE forward model

## Formatting

Code formatting and import sorting is done automatically with `black` and `isort`.

Fix import and format : `bash fixFormat.sh`

Pushes and PR to `main` branch are forbidden without first running these commands

## References

SWR-22-12



```

@conference{hassanaly2022physics,
  title={Physics-Informed Neural Network Modeling of Li-Ion Batteries},
  author={Hassanaly, Malik and Weddle, Peter and Smith, Kandler and De, Subhayan and Doostan, Alireza and King, Ryan N},
  year={2022},
  institution={National Renewable Energy Lab.(NREL), Golden, CO (United States)}
}

```




